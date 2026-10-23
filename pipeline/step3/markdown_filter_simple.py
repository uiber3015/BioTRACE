"""
Step 3: Simple Markdown Table Filtering with Qwen72b
Filters markdown tables based on TWO criteria:
1. Is it a real table (not pseudocode)?
2. Is it a data table (not terminology definitions)?
"""
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class SimpleMarkdownFilter:
    """Main class for simple markdown table filtering (2 conditions only)"""
    
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
        
        # Storage for results (thread-safe with lock)
        self.filter_results = {}
        self.table_contents = {}  # 存储表格的完整内容
        self.statistics = {
            "total_tables": 0,
            "processed_tables": 0,
            "passed_tables": 0,
            "failed_tables": 0,
            "failure_reasons": defaultdict(int),
            "success_count": 0,
            "error_count": 0
        }
        
        # Thread safety
        self.results_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.processed_count = 0  # Thread-safe counter
        self.processed_lock = threading.Lock()
    
    def run(self, max_tables: int = None):
        """Main execution method
        
        Args:
            max_tables: Maximum number of tables to process (for testing). None = process all.
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Simple Markdown Table Filtering (2 conditions)")
        self.logger.info("=" * 80)
        sys.stdout.flush()  # Force flush for nohup visibility
        
        # Load existing results for resume
        self._load_existing_results()
        
        # Get all markdown files
        input_path = Path(cfg.INPUT_DIR)
        if not input_path.exists():
            self.logger.error(f"Input directory not found: {cfg.INPUT_DIR}")
            sys.stdout.flush()
            return
        
        markdown_files = list(input_path.glob("*.md"))
        
        # Limit files if max_tables specified (testing mode)
        if max_tables is not None and max_tables > 0:
            markdown_files = markdown_files[:max_tables]
            self.logger.info(f"[TEST MODE] Limiting to {max_tables} tables")
            sys.stdout.flush()
        
        # Filter out already processed tables
        already_processed = set(self.filter_results.keys())
        if already_processed:
            original_count = len(markdown_files)
            markdown_files = [f for f in markdown_files 
                            if extract_table_id_from_filename(str(f)) not in already_processed]
            skipped = original_count - len(markdown_files)
            self.logger.info(f"Resuming: {skipped} tables already processed, {len(markdown_files)} remaining")
            sys.stdout.flush()
        
        # Total tables = already processed + remaining
        self.statistics["total_tables"] = len(already_processed) + len(markdown_files)
        
        self.logger.info(f"Found {len(markdown_files)} markdown files to process")
        self.logger.info("Filtering criteria: Real Table + Data Table")
        self.logger.info(f"Using {cfg.MAX_WORKERS} concurrent workers")
        sys.stdout.flush()  # Force flush for nohup visibility
        
        # Process each markdown file with concurrency
        self._process_tables(markdown_files)
        
        # Calculate final statistics
        self._calculate_final_statistics()
        
        # Save results
        self._save_results()
        
        self.logger.info("=" * 80)
        self.logger.info("Simple filtering completed successfully!")
        self.logger.info("=" * 80)
        sys.stdout.flush()  # Force flush for nohup visibility
    
    def _process_tables(self, markdown_files: List[Path]):
        """Process each markdown table with concurrent workers"""
        total = len(markdown_files)
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_table, md_file, idx, total): md_file
                for idx, md_file in enumerate(markdown_files, 1)
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                completed += 1
                md_file = future_to_file[future]
                
                try:
                    future.result()  # This will raise any exception that occurred
                except Exception as e:
                    table_id = extract_table_id_from_filename(str(md_file))
                    self.logger.error(f"  ✗ Unexpected error in thread for {table_id}: {e}")
                
                # Save progress periodically
                with self.processed_lock:
                    if completed % cfg.SAVE_PROGRESS_EVERY == 0:
                        self._save_intermediate_results()
                        self.logger.info(f"Progress saved at {completed}/{total}")
                        sys.stdout.flush()  # Force flush for nohup visibility
        
        # Final save
        self._save_intermediate_results()
        sys.stdout.flush()
    
    def _process_single_table(self, md_file: Path, idx: int, total: int):
        """Process a single markdown table (thread-safe)"""
        table_id = extract_table_id_from_filename(str(md_file))
        
        # Log progress (every 100 tables)
        with self.processed_lock:
            current_processed = self.processed_count + 1
            if current_processed % 100 == 0:
                self.logger.info(f"Processing [{current_processed}/{total}] {table_id}")
                sys.stdout.flush()  # Force flush for nohup visibility
            else:
                self.logger.debug(f"Processing [{current_processed}/{total}] {table_id}")
        
        try:
            # Load markdown content
            markdown_content = load_markdown(str(md_file))
            
            # Check if content is empty or too short
            if not markdown_content or len(markdown_content.strip()) < 10:
                self.logger.warning(f"  ⚠ Empty or too short content: {table_id}")
                with self.results_lock:
                    self.filter_results[table_id] = {
                        "is_real_table": False,
                        "is_data_table": False,
                        "reason": "Empty or invalid content",
                        "passed": False,
                        "status": "skipped"
                    }
                with self.stats_lock:
                    self.statistics["failed_tables"] += 1
                    self.statistics["processed_tables"] += 1
                with self.processed_lock:
                    self.processed_count += 1
                return
            
            # Extract relevant sections (Caption + Table Content)
            extracted_content = extract_relevant_sections(markdown_content)
            
            # Check if extracted content is empty
            if not extracted_content or len(extracted_content.strip()) < 10:
                self.logger.warning(f"  ⚠ No table content found: {table_id}")
                with self.results_lock:
                    self.filter_results[table_id] = {
                        "is_real_table": False,
                        "is_data_table": False,
                        "reason": "No table content found",
                        "passed": False,
                        "status": "skipped"
                    }
                with self.stats_lock:
                    self.statistics["failed_tables"] += 1
                    self.statistics["processed_tables"] += 1
                with self.processed_lock:
                    self.processed_count += 1
                return
            
            # Smart truncation: only keep essential info for filtering judgment
            # (Caption + Header + First N sample rows)
            truncated_content = smart_truncate_for_filtering(
                extracted_content,
                sample_rows=cfg.SAMPLE_ROWS_FOR_FILTERING
            )
            
            # Fallback: further truncate if still too long (should rarely happen)
            if len(truncated_content) > cfg.MAX_CONTENT_LENGTH:
                truncated_content = truncated_content[:cfg.MAX_CONTENT_LENGTH] + "\n\n...(truncated)"
            
            # Call Qwen for filtering (2 conditions only)
            result = self.qwen_client.filter_table_simple(
                truncated_content,
                retry_times=cfg.RETRY_TIMES,
                backoff_factor=cfg.BACKOFF_FACTOR
            )
            
            if result:
                # Validate result consistency
                is_real = result.get("is_real_table", False)
                is_data = result.get("is_data_table", False)
                reason = result.get("reason", "")
                
                # Check for logical inconsistencies
                if not is_real and not is_data and ("数据" in reason or "实验" in reason):
                    self.logger.warning(f"  ⚠ Inconsistent result for {table_id}: {reason}")
                
                # Determine if table passed (both conditions must be True)
                passed = is_real and is_data
                
                # Thread-safe update of results and statistics
                with self.results_lock:
                    self.filter_results[table_id] = {
                        "is_real_table": is_real,
                        "is_data_table": is_data,
                        "reason": reason,
                        "passed": passed,
                        "status": "success"
                    }
                    # 如果表格通过筛选，保存其完整内容
                    if passed:
                        self.table_contents[table_id] = {
                            "markdown_content": markdown_content,
                            "file_path": str(md_file)
                        }
                
                with self.stats_lock:
                    if passed:
                        self.statistics["passed_tables"] += 1
                    else:
                        self.statistics["failed_tables"] += 1
                        self._record_failure_reason(result)
                    self.statistics["success_count"] += 1
                    self.statistics["processed_tables"] += 1
                
                with self.processed_lock:
                    current_processed = self.processed_count + 1
                    if current_processed % 100 == 0:
                        self.logger.info(f"  ✓ Result: {'PASS' if passed else 'FAIL'}")
                        sys.stdout.flush()  # Force flush for nohup visibility
                    self.processed_count += 1
            else:
                with self.results_lock:
                    self.filter_results[table_id] = {
                        "is_real_table": False,
                        "is_data_table": False,
                        "reason": "API call failed",
                        "passed": False,
                        "status": "failed"
                    }
                with self.stats_lock:
                    self.statistics["error_count"] += 1
                    self.statistics["failed_tables"] += 1
                    self.statistics["processed_tables"] += 1
                with self.processed_lock:
                    current_processed = self.processed_count + 1
                    if current_processed % 100 == 0:
                        self.logger.warning(f"  ✗ Analysis failed")
                        sys.stdout.flush()  # Force flush for nohup visibility
                    self.processed_count += 1
            
            # Small delay to avoid overwhelming the API server
            # (Even with concurrency, we want some rate limiting)
            time.sleep(cfg.REQUEST_DELAY)
            
        except Exception as e:
            self.logger.error(f"  ✗ Error processing {table_id}: {e}")
            sys.stdout.flush()  # Force flush for nohup visibility
            with self.results_lock:
                self.filter_results[table_id] = {
                    "is_real_table": False,
                    "is_data_table": False,
                    "reason": f"Error: {str(e)}",
                    "passed": False,
                    "status": "error"
                }
            with self.stats_lock:
                self.statistics["error_count"] += 1
                self.statistics["failed_tables"] += 1
                self.statistics["processed_tables"] += 1
            with self.processed_lock:
                self.processed_count += 1
    
    def _record_failure_reason(self, result: Dict):
        """Record why a table failed filtering (called within stats_lock)"""
        if not result.get("is_real_table", False):
            self.statistics["failure_reasons"]["not_real_table"] += 1
        if not result.get("is_data_table", False):
            self.statistics["failure_reasons"]["not_data_table"] += 1
    
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
            self.output_dir / "simple_filter_results.json"
        )
        
        # Save filtered table list (only passed tables)
        filtered_tables = [
            table_id for table_id, result in self.filter_results.items()
            if result.get("passed", False)
        ]
        save_json(
            {"filtered_tables": filtered_tables, "count": len(filtered_tables)},
            self.output_dir / "simple_filtered_tables.json"
        )
        
        # Save table contents (通过筛选的表格的完整内容)
        save_json(
            self.table_contents,
            self.output_dir / "simple_filtered_tables_with_content.json"
        )
        
        # Save statistics
        save_json(
            dict(self.statistics),
            self.output_dir / "simple_filter_statistics.json"
        )
        
        self.logger.info(f"\nResults saved to {self.output_dir}")
        self.logger.info(f"  - simple_filter_results.json: Detailed results")
        self.logger.info(f"  - simple_filtered_tables.json: List of {len(filtered_tables)} passed tables")
        self.logger.info(f"  - simple_filtered_tables_with_content.json: {len(self.table_contents)} tables with full content")
        self.logger.info(f"  - simple_filter_statistics.json: Statistics")
    
    def _load_existing_results(self):
        """Load existing results for resume functionality"""
        results_file = self.output_dir / "simple_filter_results.json"
        contents_file = self.output_dir / "simple_filtered_tables_with_content.json"
        
        if results_file.exists():
            try:
                existing_results = load_json(str(results_file))
                self.filter_results = existing_results
                
                # Load existing table contents if available
                if contents_file.exists():
                    try:
                        existing_contents = load_json(str(contents_file))
                        self.table_contents = existing_contents
                        self.logger.info(f"Loaded {len(existing_contents)} existing table contents")
                    except Exception as e:
                        self.logger.warning(f"Failed to load existing contents: {e}")
                        self.table_contents = {}
                
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
                self.table_contents = {}
    
    def _save_intermediate_results(self):
        """Save intermediate results (thread-safe)"""
        with self.results_lock:
            # Create a copy to avoid holding lock during I/O
            results_copy = dict(self.filter_results)
            contents_copy = dict(self.table_contents)
        save_json(
            results_copy,
            self.output_dir / "simple_filter_results.json"
        )
        # Also save table contents
        if contents_copy:
            save_json(
                contents_copy,
                self.output_dir / "simple_filtered_tables_with_content.json"
            )


def main():
    """Main entry point"""
    filter_system = SimpleMarkdownFilter()
    # Use SAMPLE_SIZE from config (None = process all)
    max_tables = getattr(cfg, 'SAMPLE_SIZE', None)
    filter_system.run(max_tables=max_tables)


if __name__ == "__main__":
    main()

