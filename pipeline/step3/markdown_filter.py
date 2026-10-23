"""
Step 3: Markdown Table Filtering with Qwen72b
Filters markdown tables based on three criteria using Qwen text analysis
"""
import sys
import time
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from qwen_client import QwenClient
from utils import (
    setup_logging,
    ensure_dir,
    load_json,
    save_json,
    load_markdown,
    extract_table_id_from_filename,
    extract_relevant_sections,
    smart_truncate_for_filtering
)


class MarkdownFilter:
    """Main class for markdown table filtering"""
    
    def __init__(self):
        self.logger = setup_logging(cfg.LOG_DIR)
        self.output_dir = ensure_dir(cfg.OUTPUT_DIR)
        
        # Initialize Qwen client
        self.qwen_client = QwenClient(
            api_url=cfg.QWEN_API_URL,
            model=cfg.QWEN_MODEL,
            temperature=cfg.QWEN_TEMPERATURE,
            timeout=cfg.TIMEOUT,
            use_openai_format=cfg.USE_OPENAI_FORMAT,
            api_key=cfg.QWEN_API_KEY
        )
        
        # Load entity types from step2
        self.entity_types = self._load_entity_types()
        
        # Storage for results
        self.filter_results = {}
        self.statistics = {
            "total_tables": 0,
            "processed_tables": 0,
            "passed_tables": 0,
            "failed_tables": 0,
            "failure_reasons": defaultdict(int),
            "success_count": 0,
            "error_count": 0
        }
    
    def _load_entity_types(self) -> List[str]:
        """Load 80th percentile entity types from step2 output"""
        try:
            data = load_json(cfg.ENTITY_TYPES_FILE)
            entity_types = data.get("selected_entity_types", [])
            self.logger.info(f"Loaded {len(entity_types)} entity types from step2")
            self.logger.info(f"Entity types: {', '.join(entity_types[:10])}...")
            return entity_types
        except Exception as e:
            self.logger.error(f"Failed to load entity types: {e}")
            self.logger.warning("Using default entity types")
            return [
                "Gene", "Protein", "Disease", "Drug", "Cell Type",
                "Tissue/Organ", "Pathway", "Phenotype", "Compound"
            ]
    
    def run(self, max_tables: int = None):
        """Main execution method
        
        Args:
            max_tables: Maximum number of tables to process (for testing). None = process all.
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Step 3: Markdown Table Filtering (3 conditions)")
        self.logger.info("=" * 80)
        
        # Load existing results for resume
        self._load_existing_results()
        
        # Get all markdown files
        input_path = Path(cfg.INPUT_DIR)
        if not input_path.exists():
            self.logger.error(f"Input directory not found: {cfg.INPUT_DIR}")
            return
        
        markdown_files = list(input_path.glob("*.md"))
        
        # Limit files if max_tables specified (testing mode)
        if max_tables is not None and max_tables > 0:
            markdown_files = markdown_files[:max_tables]
            self.logger.info(f"[TEST MODE] Limiting to {max_tables} tables")
        
        # Filter out already processed tables
        already_processed = set(self.filter_results.keys())
        if already_processed:
            original_count = len(markdown_files)
            markdown_files = [f for f in markdown_files 
                            if extract_table_id_from_filename(str(f)) not in already_processed]
            skipped = original_count - len(markdown_files)
            self.logger.info(f"Resuming: {skipped} tables already processed, {len(markdown_files)} remaining")
        
        # Total tables = already processed + remaining
        self.statistics["total_tables"] = len(already_processed) + len(markdown_files)
        
        self.logger.info(f"Found {len(markdown_files)} markdown files to process")
        self.logger.info(f"Using {len(self.entity_types)} entity types for filtering")
        self.logger.info("Filtering criteria: Real Table + Data Table + Contains Entities")
        
        # Process each markdown file
        self._process_tables(markdown_files)
        
        # Calculate final statistics
        self._calculate_final_statistics()
        
        # Save results
        self._save_results()
        
        self.logger.info("=" * 80)
        self.logger.info("Step 3 completed successfully!")
        self.logger.info("=" * 80)
    
    def _process_tables(self, markdown_files: List[Path]):
        """Process each markdown table"""
        total = len(markdown_files)
        
        for idx, md_file in enumerate(markdown_files, 1):
            table_id = extract_table_id_from_filename(str(md_file))
            
            if idx % 100 == 0:
                self.logger.info(f"Processing [{idx}/{total}] {table_id}")
            else:
                self.logger.debug(f"Processing [{idx}/{total}] {table_id}")
            
            try:
                # Load markdown content
                markdown_content = load_markdown(str(md_file))
                
                # Check if content is empty or too short
                if not markdown_content or len(markdown_content.strip()) < 10:
                    self.logger.warning(f"  ⚠ Empty or too short content: {table_id}")
                    self.filter_results[table_id] = {
                        "is_real_table": False,
                        "is_data_table": False,
                        "contains_entities": False,
                        "found_entity_types": [],
                        "reason": "Empty or invalid content",
                        "passed": False,
                        "status": "skipped"
                    }
                    self.statistics["failed_tables"] += 1
                    self.statistics["processed_tables"] += 1
                    continue
                
                # Extract relevant sections (Caption + Table Content)
                extracted_content = extract_relevant_sections(markdown_content)
                
                # Check if extracted content is empty
                if not extracted_content or len(extracted_content.strip()) < 10:
                    self.logger.warning(f"  ⚠ No table content found: {table_id}")
                    self.filter_results[table_id] = {
                        "is_real_table": False,
                        "is_data_table": False,
                        "contains_entities": False,
                        "found_entity_types": [],
                        "reason": "No table content found",
                        "passed": False,
                        "status": "skipped"
                    }
                    self.statistics["failed_tables"] += 1
                    self.statistics["processed_tables"] += 1
                    continue
                
                # Smart truncation: only keep essential info for filtering judgment
                # (Caption + Header + First N sample rows)
                truncated_content = smart_truncate_for_filtering(
                    extracted_content,
                    sample_rows=cfg.SAMPLE_ROWS_FOR_FILTERING
                )
                
                # Fallback: further truncate if still too long (should rarely happen)
                if len(truncated_content) > cfg.MAX_CONTENT_LENGTH:
                    truncated_content = truncated_content[:cfg.MAX_CONTENT_LENGTH] + "\n\n...(truncated)"
                
                # Call Qwen for filtering
                result = self.qwen_client.filter_table(
                    truncated_content,
                    self.entity_types,
                    retry_times=cfg.RETRY_TIMES,
                    backoff_factor=cfg.BACKOFF_FACTOR
                )
                
                if result:
                    # Determine if table passed
                    passed = self._evaluate_filter_result(result)
                    
                    self.filter_results[table_id] = {
                        "is_real_table": result.get("is_real_table", False),
                        "is_data_table": result.get("is_data_table", False),
                        "contains_entities": result.get("contains_entities", False),
                        "found_entity_types": result.get("found_entity_types", []),
                        "reason": result.get("reason", ""),
                        "passed": passed,
                        "status": "success"
                    }
                    
                    if passed:
                        self.statistics["passed_tables"] += 1
                    else:
                        self.statistics["failed_tables"] += 1
                        self._record_failure_reason(result)
                    
                    self.statistics["success_count"] += 1
                    
                    if idx % 100 == 0:
                        self.logger.info(f"  ✓ Result: {'PASS' if passed else 'FAIL'}")
                else:
                    self.filter_results[table_id] = {
                        "is_real_table": False,
                        "is_data_table": False,
                        "contains_entities": False,
                        "found_entity_types": [],
                        "reason": "API call failed",
                        "passed": False,
                        "status": "failed"
                    }
                    self.statistics["error_count"] += 1
                    self.statistics["failed_tables"] += 1
                    
                    if idx % 100 == 0:
                        self.logger.warning(f"  ✗ Analysis failed")
                
                self.statistics["processed_tables"] += 1
                
            except Exception as e:
                self.logger.error(f"  ✗ Error processing {table_id}: {e}")
                self.filter_results[table_id] = {
                    "is_real_table": False,
                    "is_data_table": False,
                    "contains_entities": False,
                    "found_entity_types": [],
                    "reason": f"Error: {str(e)}",
                    "passed": False,
                    "status": "error"
                }
                self.statistics["error_count"] += 1
                self.statistics["failed_tables"] += 1
            
            # Save progress periodically
            if idx % cfg.SAVE_PROGRESS_EVERY == 0:
                self._save_intermediate_results()
                self.logger.info(f"Progress saved at {idx}/{total}")
            
            # Delay between requests
            time.sleep(cfg.REQUEST_DELAY)
        
        # Final save
        self._save_intermediate_results()
    
    def _evaluate_filter_result(self, result: Dict) -> bool:
        """Evaluate if table passes filtering criteria"""
        if cfg.REQUIRE_ALL_CONDITIONS:
            # All three conditions must be True
            return (
                result.get("is_real_table", False) and
                result.get("is_data_table", False) and
                result.get("contains_entities", False)
            )
        else:
            # At least one condition must be True
            return (
                result.get("is_real_table", False) or
                result.get("is_data_table", False) or
                result.get("contains_entities", False)
            )
    
    def _record_failure_reason(self, result: Dict):
        """Record why a table failed filtering"""
        if not result.get("is_real_table", False):
            self.statistics["failure_reasons"]["not_real_table"] += 1
        if not result.get("is_data_table", False):
            self.statistics["failure_reasons"]["not_data_table"] += 1
        if not result.get("contains_entities", False):
            self.statistics["failure_reasons"]["no_entities"] += 1
    
    def _calculate_final_statistics(self):
        """Calculate final statistics"""
        if self.statistics["processed_tables"] > 0:
            self.statistics["pass_rate"] = (
                self.statistics["passed_tables"] / self.statistics["processed_tables"]
            )
        else:
            self.statistics["pass_rate"] = 0.0
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FILTERING STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Total tables: {self.statistics['total_tables']}")
        self.logger.info(f"Processed: {self.statistics['processed_tables']}")
        self.logger.info(f"Passed: {self.statistics['passed_tables']}")
        self.logger.info(f"Failed: {self.statistics['failed_tables']}")
        self.logger.info(f"Pass rate: {self.statistics['pass_rate']:.2%}")
        self.logger.info(f"Errors: {self.statistics['error_count']}")
        self.logger.info("\nFailure reasons:")
        for reason, count in self.statistics["failure_reasons"].items():
            self.logger.info(f"  {reason}: {count}")
    
    def _save_results(self):
        """Save final results"""
        # Save filter results
        save_json(
            self.filter_results,
            self.output_dir / "filter_results.json"
        )
        
        # Save filtered table list (only passed tables)
        filtered_tables = [
            table_id for table_id, result in self.filter_results.items()
            if result.get("passed", False)
        ]
        save_json(
            {"filtered_tables": filtered_tables, "count": len(filtered_tables)},
            self.output_dir / "filtered_tables.json"
        )
        
        # Save statistics
        save_json(
            dict(self.statistics),
            self.output_dir / "filter_statistics.json"
        )
        
        self.logger.info(f"\nResults saved to {self.output_dir}")
        self.logger.info(f"  - filter_results.json: Detailed results")
        self.logger.info(f"  - filtered_tables.json: List of {len(filtered_tables)} passed tables")
        self.logger.info(f"  - filter_statistics.json: Statistics")
    
    def _load_existing_results(self):
        """Load existing results for resume functionality"""
        results_file = self.output_dir / "filter_results.json"
        if results_file.exists():
            try:
                existing_results = load_json(str(results_file))
                self.filter_results = existing_results
                # Recalculate statistics from existing results
                for result in existing_results.values():
                    if result.get("status") == "success":
                        self.statistics["success_count"] += 1
                        if result.get("passed", False):
                            self.statistics["passed_tables"] += 1
                        else:
                            self.statistics["failed_tables"] += 1
                            self._record_failure_reason(result)
                    else:
                        self.statistics["error_count"] += 1
                        self.statistics["failed_tables"] += 1
                self.statistics["processed_tables"] = len(existing_results)
                self.logger.info(f"Loaded {len(existing_results)} existing results")
            except Exception as e:
                self.logger.warning(f"Failed to load existing results: {e}, starting fresh")
                self.filter_results = {}
    
    def _save_intermediate_results(self):
        """Save intermediate results"""
        save_json(
            self.filter_results,
            self.output_dir / "filter_results.json"
        )


def main():
    """Main entry point"""
    filter_system = MarkdownFilter()
    # Use SAMPLE_SIZE from config (None = process all)
    max_tables = getattr(cfg, 'SAMPLE_SIZE', None)
    filter_system.run(max_tables=max_tables)


if __name__ == "__main__":
    main()

