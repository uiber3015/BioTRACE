"""
Step 2: Entity Sampling and Statistical Analysis
Randomly samples markdown tables and analyzes entity types using GPT-5
"""
import sys
import time
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add step2 directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from gpt5_client import GPT5Client
from utils import (
    setup_logging,
    ensure_dir,
    load_json,
    save_json,
    load_markdown,
    extract_table_content,
    sample_markdown_files,
    extract_table_id_from_filename
)


class EntitySampler:
    """Main class for entity sampling and analysis"""
    
    def __init__(self):
        self.logger = setup_logging(cfg.LOG_DIR)
        self.output_dir = ensure_dir(cfg.OUTPUT_DIR)
        
        # Initialize GPT-5 client
        self.gpt5_client = GPT5Client(
            api_url=cfg.GPT5_API_URL,
            api_key=cfg.GPT5_API_KEY,
            model=cfg.GPT5_MODEL,
            temperature=cfg.GPT5_TEMPERATURE,
            max_tokens=cfg.GPT5_MAX_TOKENS,
            timeout=cfg.TIMEOUT
        )
        
        # Storage for results
        self.sampled_tables = []
        self.entity_analysis = {}
        self.entity_statistics = defaultdict(lambda: {"table_count": 0, "cell_count": 0})
        
        # Thread safety
        self.lock = Lock()
        self.processed_count = 0
        
    def run(self):
        """Main execution method"""
        self.logger.info("=" * 80)
        self.logger.info("Starting Step 2: Entity Sampling and Analysis")
        self.logger.info("=" * 80)
        
        # Step 1: Sample markdown files
        self.logger.info(f"Sampling {cfg.SAMPLE_SIZE} markdown files from {cfg.INPUT_DIR}")
        try:
            self.sampled_tables = sample_markdown_files(
                cfg.INPUT_DIR,
                cfg.SAMPLE_SIZE,
                cfg.RANDOM_SEED
            )
            self.logger.info(f"Successfully sampled {len(self.sampled_tables)} files")
        except Exception as e:
            self.logger.error(f"Failed to sample files: {e}")
            return
        
        # Save sampled table list
        sampled_ids = [extract_table_id_from_filename(f) for f in self.sampled_tables]
        save_json(
            {"sampled_tables": sampled_ids, "count": len(sampled_ids)},
            self.output_dir / "sampled_tables.json"
        )
        
        # Step 2: Analyze each sampled table
        self.logger.info("Starting entity analysis...")
        self._analyze_tables()
        
        # Step 3: Calculate statistics
        self.logger.info("Calculating entity statistics...")
        self._calculate_statistics()
        
        # Step 4: Calculate 80th percentile
        self.logger.info("Calculating 80th percentile entity types...")
        self._calculate_percentile()
        
        self.logger.info("=" * 80)
        self.logger.info("Step 2 completed successfully!")
        self.logger.info("=" * 80)
    
    def _analyze_single_table(self, table_path: Path, idx: int, total: int) -> tuple:
        """Analyze a single table (thread-safe)"""
        table_id = extract_table_id_from_filename(table_path)
        
        try:
            # Load markdown content
            markdown_content = load_markdown(table_path)
            
            # Extract relevant sections
            table_content = extract_table_content(markdown_content)
            
            # Call GPT-5 for entity analysis
            result = self.gpt5_client.analyze_table_entities(
                table_content,
                retry_times=cfg.RETRY_TIMES,
                backoff_factor=cfg.BACKOFF_FACTOR
            )
            
            if result:
                analysis = {
                    "entity_types": result.get("entity_types", []),
                    "entity_counts": result.get("entity_counts", {}),
                    "status": "success"
                }
                self.logger.info(f"  ✓ [{idx}/{total}] {table_id}: Found {len(result.get('entity_types', []))} entity types")
                return table_id, analysis, True
            else:
                analysis = {
                    "entity_types": [],
                    "entity_counts": {},
                    "status": "failed"
                }
                self.logger.warning(f"  ✗ [{idx}/{total}] {table_id}: Analysis failed")
                return table_id, analysis, False
                
        except Exception as e:
            self.logger.error(f"  ✗ [{idx}/{total}] {table_id}: Error - {e}")
            analysis = {
                "entity_types": [],
                "entity_counts": {},
                "status": "error",
                "error": str(e)
            }
            return table_id, analysis, False
    
    def _analyze_tables(self):
        """Analyze each sampled table for entity types (with parallel processing)"""
        # 🆕 断点续传：加载已有进度
        progress_file = self.output_dir / "entity_analysis.json"
        if progress_file.exists():
            try:
                existing_data = load_json(progress_file)
                self.entity_analysis = existing_data
                already_processed = set(existing_data.keys())
                self.logger.info(f"✓ 发现已有进度: {len(already_processed)} 个表格已处理")
                
                # 过滤出未处理的表格
                remaining_tables = []
                for table_path in self.sampled_tables:
                    table_id = extract_table_id_from_filename(table_path)
                    if table_id not in already_processed:
                        remaining_tables.append(table_path)
                
                self.sampled_tables = remaining_tables
                
                # 重建统计信息
                for table_id, analysis in existing_data.items():
                    if analysis.get("status") == "success":
                        for entity_type in analysis.get("entity_types", []):
                            self.entity_statistics[entity_type]["table_count"] += 1
                            cell_count = analysis.get("entity_counts", {}).get(entity_type, 0)
                            self.entity_statistics[entity_type]["cell_count"] += cell_count
                
                self.logger.info(f"✓ 剩余待处理: {len(self.sampled_tables)} 个表格")
            except Exception as e:
                self.logger.warning(f"⚠ 无法加载进度文件: {e}，将从头开始")
        
        total = len(self.sampled_tables)
        
        if total == 0:
            self.logger.info("=" * 80)
            self.logger.info("✓ 所有表格已处理完成！跳过分析阶段")
            self.logger.info("=" * 80)
            return
        
        self.logger.info(f"开始处理剩余 {total} 个表格...")
        
        if cfg.USE_PARALLEL:
            self.logger.info(f"Using parallel processing with {cfg.MAX_WORKERS} workers")
            self._analyze_tables_parallel(total)
        else:
            self.logger.info("Using sequential processing")
            self._analyze_tables_sequential(total)
        
        # Final save
        self._save_intermediate_results()
        
        # Print final statistics
        success_count = sum(1 for a in self.entity_analysis.values() if a.get('status') == 'success')
        fail_count = len(self.entity_analysis) - success_count
        success_rate = (success_count / len(self.entity_analysis) * 100) if len(self.entity_analysis) > 0 else 0
        
        self.logger.info("=" * 80)
        self.logger.info("📊 Final Analysis Statistics:")
        self.logger.info(f"   Total Processed: {len(self.entity_analysis)}")
        self.logger.info(f"   ✓ Successful: {success_count} ({success_rate:.1f}%)")
        self.logger.info(f"   ✗ Failed: {fail_count} ({100-success_rate:.1f}%)")
        self.logger.info("=" * 80)
    
    def _analyze_tables_parallel(self, total: int):
        """Parallel analysis using ThreadPoolExecutor with rate limiting"""
        with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
            futures = []
            
            # Submit tasks with delay to avoid API burst
            for idx, table_path in enumerate(self.sampled_tables, 1):
                future = executor.submit(self._analyze_single_table, table_path, idx, total)
                futures.append((future, idx, table_path))
                
                # Add delay between submissions to stagger requests
                if idx < len(self.sampled_tables):
                    time.sleep(cfg.REQUEST_DELAY)
            
            # Process completed tasks
            for future, idx, table_path in futures:
                try:
                    table_id, analysis, success = future.result()
                    
                    # Thread-safe update
                    with self.lock:
                        self.entity_analysis[table_id] = analysis
                        
                        # Update statistics if successful
                        if success:
                            for entity_type in analysis.get("entity_types", []):
                                self.entity_statistics[entity_type]["table_count"] += 1
                                cell_count = analysis.get("entity_counts", {}).get(entity_type, 0)
                                self.entity_statistics[entity_type]["cell_count"] += cell_count
                        
                        self.processed_count += 1
                        
                        # Save progress periodically
                        if self.processed_count % cfg.SAVE_PROGRESS_EVERY == 0:
                            self._save_intermediate_results()
                            success_count = sum(1 for a in self.entity_analysis.values() if a.get('status') == 'success')
                            fail_count = self.processed_count - success_count
                            success_rate = (success_count / self.processed_count * 100) if self.processed_count > 0 else 0
                            self.logger.info(f"📊 Progress: {self.processed_count}/{total} | ✓ {success_count} | ✗ {fail_count} | Success Rate: {success_rate:.1f}%")
                
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {e}")
    
    def _analyze_tables_sequential(self, total: int):
        """Sequential analysis (original method)"""
        for idx, table_path in enumerate(self.sampled_tables, 1):
            table_id, analysis, success = self._analyze_single_table(table_path, idx, total)
            
            self.entity_analysis[table_id] = analysis
            
            # Update statistics if successful
            if success:
                for entity_type in analysis.get("entity_types", []):
                    self.entity_statistics[entity_type]["table_count"] += 1
                    cell_count = analysis.get("entity_counts", {}).get(entity_type, 0)
                    self.entity_statistics[entity_type]["cell_count"] += cell_count
            
            # Save progress periodically
            if idx % cfg.SAVE_PROGRESS_EVERY == 0:
                self._save_intermediate_results()
                self.logger.info(f"Progress saved at {idx}/{total}")
            
            # Delay between requests
            time.sleep(cfg.REQUEST_DELAY)
    
    def _calculate_statistics(self):
        """Calculate entity type statistics"""
        stats = {
            "total_sampled_tables": len(self.sampled_tables),
            "successfully_analyzed": sum(
                1 for v in self.entity_analysis.values() if v["status"] == "success"
            ),
            "failed_analysis": sum(
                1 for v in self.entity_analysis.values() if v["status"] != "success"
            ),
            "entity_type_frequency": dict(self.entity_statistics),
            "total_unique_entity_types": len(self.entity_statistics)
        }
        
        # Sort by table_count descending
        sorted_entities = sorted(
            stats["entity_type_frequency"].items(),
            key=lambda x: x[1]["table_count"],
            reverse=True
        )
        stats["entity_type_frequency_sorted"] = [
            {"entity_type": k, **v} for k, v in sorted_entities
        ]
        
        save_json(stats, self.output_dir / "entity_statistics.json")
        
        # Log top 10 entity types
        self.logger.info("\nTop 10 Entity Types:")
        for i, (entity_type, counts) in enumerate(sorted_entities[:10], 1):
            self.logger.info(
                f"  {i}. {entity_type}: "
                f"{counts['table_count']} tables, "
                f"{counts['cell_count']} cells"
            )
    
    def _calculate_percentile(self):
        """Select minimal entity set to cover at least P% of tables (greedy set cover)"""
        import math
        
        def normalize_entity(e: str) -> str:
            """Title Case normalization to avoid case sensitivity issues"""
            return (e or "").strip().title()
        
        # Only count successfully analyzed tables
        success_items = {
            tid: v for tid, v in self.entity_analysis.items() 
            if v["status"] == "success"
        }
        total_tables = len(success_items)
        
        if total_tables == 0:
            self.logger.error("No successfully analyzed tables!")
            return
        
        threshold = math.ceil(total_tables * (cfg.PERCENTILE_THRESHOLD / 100.0))
        
        # Build coverage mapping: entity -> set(tables), and accumulate tie-breaker metrics
        entity_to_tables = defaultdict(set)
        entity_cell_sum = defaultdict(int)
        
        for table_id, analysis in success_items.items():
            # Table-level entity set (deduplicated)
            ents = set(normalize_entity(e) for e in analysis.get("entity_types", []))
            for e in ents:
                entity_to_tables[e].add(table_id)
            
            # Accumulate cell_count for tie-breaking
            for e_raw, cnt in (analysis.get("entity_counts", {}) or {}).items():
                e = normalize_entity(e_raw)
                try:
                    entity_cell_sum[e] += int(cnt)
                except Exception:
                    pass
        
        # Calculate table_count based on coverage mapping
        entity_table_count = {e: len(tset) for e, tset in entity_to_tables.items()}
        
        # Greedy selection: each round selects the entity with maximum new coverage gain
        # Tie-breaking by table_count, then cell_count
        covered_tables = set()
        selected_entities = []
        remaining = set(entity_to_tables.keys())
        
        while len(covered_tables) < threshold and remaining:
            def key_fn(e):
                new_gain = len(entity_to_tables[e] - covered_tables)
                return (new_gain, entity_table_count.get(e, 0), entity_cell_sum.get(e, 0))
            
            best_entity = max(remaining, key=key_fn)
            new_gain = len(entity_to_tables[best_entity] - covered_tables)
            
            if new_gain == 0:
                # Cannot improve coverage further, exit
                break
            
            selected_entities.append(best_entity)
            covered_tables.update(entity_to_tables[best_entity])
            remaining.remove(best_entity)
            
            self.logger.info(
                f"  Selected {best_entity}: +{new_gain} tables, "
                f"coverage {len(covered_tables)}/{total_tables} "
                f"({len(covered_tables) / total_tables * 100:.1f}%)"
            )
        
        result = {
            "percentile": cfg.PERCENTILE_THRESHOLD,
            "total_tables_analyzed": total_tables,
            "threshold_tables": threshold,
            "actual_covered_tables": len(covered_tables),
            "coverage_percentage": (len(covered_tables) / total_tables * 100.0),
            "selected_entity_types": selected_entities,
            "total_entity_types_selected": len(selected_entities),
            "entity_details": {
                e: {
                    "table_count": entity_table_count.get(e, 0),
                    "cell_count": entity_cell_sum.get(e, 0)
                }
                for e in selected_entities
            },
        }
        
        save_json(result, self.output_dir / "percentile_80_entities.json")
        
        self.logger.info("\n80th Percentile Results (Set Cover Algorithm):")
        self.logger.info(f"  Total tables: {total_tables}")
        self.logger.info(f"  Target coverage: {threshold} tables ({cfg.PERCENTILE_THRESHOLD}%)")
        self.logger.info(f"  Actual coverage: {len(covered_tables)} tables ({result['coverage_percentage']:.1f}%)")
        self.logger.info(f"  Entity types selected: {len(selected_entities)}")
        self.logger.info(f"\nSelected Entity Types:")
        for entity in selected_entities:
            freq = entity_table_count.get(entity, 0)
            cells = entity_cell_sum.get(entity, 0)
            self.logger.info(f"  - {entity}: {freq} tables, {cells} cells")
    
    def _save_intermediate_results(self):
        """Save intermediate results"""
        save_json(self.entity_analysis, self.output_dir / "entity_analysis.json")


def main():
    """Main entry point"""
    sampler = EntitySampler()
    sampler.run()


if __name__ == "__main__":
    main()

