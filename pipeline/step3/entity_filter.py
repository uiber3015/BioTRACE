"""
Step 3: Entity Type Filtering
基于 markdown_filter_simple.py 的结果，对已通过前两个条件的表格进行实体类型筛选
分为三类：全部在、部分在、全都不在
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


class EntityFilter:
    """对已通过前两个条件的表格进行实体类型筛选"""
    
    def __init__(self):
        self.logger = setup_logging(cfg.LOG_DIR, log_file="entity_filter.log")
        self.output_dir = ensure_dir(cfg.OUTPUT_DIR)
        
        # 加载14个实体类型（带说明）
        self.entity_types = self._load_entity_types()
        
        # 初始化 Qwen client
        self.qwen_client = QwenClient(
            api_url=cfg.QWEN_API_URL,
            model=cfg.QWEN_MODEL,
            temperature=cfg.QWEN_TEMPERATURE,
            timeout=cfg.TIMEOUT,
            use_openai_format=cfg.USE_OPENAI_FORMAT,
            api_key=cfg.QWEN_API_KEY
        )
        
        # 存储结果
        self.entity_results = {}  # {table_id: {extracted_entities, classification, ...}}
        self.final_results = {}   # 合并后的最终结果
        
        # 统计信息
        self.statistics = {
            "input_tables": 0,              # 输入表格数（已通过前两个条件）
            "processed_tables": 0,
            "all_in": 0,                    # 全部在
            "partial_in": 0,                # 部分在
            "none_in": 0,                   # 全都不在
            "error_count": 0,
            "truncated_tables": 0,          # 被截断的表格数
            "total_original_rows": 0,       # 原始总行数
            "total_truncated_rows": 0,      # 截断后总行数
            "skipped_tables": 0,            # 跳过的表格数（文件不存在等）
            "failed_extraction": 0,         # 实体提取失败数
            "failed_classification": 0      # 实体分类失败数
        }
        
        # 线程安全
        self.results_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.processed_count = 0
        self.processed_lock = threading.Lock()
        
        # 进度统计
        self.start_time = None
        self.last_progress_time = None
        self.progress_lock = threading.Lock()
    
    def _load_entity_types(self) -> List[str]:
        """加载14个实体类型（带说明）"""
        entity_types = [
            "基因名 (Gene Symbol) - 基因的官方符号或常用名。例如：MALAT1, NEAT2, 2210401K01Rik",
            "调控因子 (Regulator) - 在调控关系中起上游作用的分子实体。例如：NONO, p53 (此类别专注于其调控功能)",
            "靶点 (Target) - 被特定分子调控的下游分子实体。例如：TP53 (此类别专注于其作为被调控目标的角色)",
            "功能机制 (Functional Mechanism) - 分子发挥功能的途径或机制。例如：转录调控, 后转录调控, 表观调控, 翻译调控, RNA, SeRNA",
            "物种 (Species) - 研究中所涉及到的生物体物种。例如：人, 小鼠, 大肠杆菌 (整合并替代原\"生物体/Organism\")",
            "疾病 (Disease) - 生物体在生理或心理上出现的异常状态，特指与研究对象（如LncRNA）关联的疾病。例如：癌症, 阿尔茨海默病",
            "器官 (Organ) - 提取所用生物材料来源的器官类型。例如：脑, 心脏, 肝脏",
            "组织 (Tissue) - 提取所用生物材料来源的组织类型。例如：皮肤组织, 心肌组织, 神经组织",
            "细胞 (Cell) - 提取所用生物材料来源的细胞系或细胞类型。例如：K562细胞, HeLa细胞, T淋巴细胞 (整合并替代原\"细胞类型/Cell Type\")",
            "蛋白质 (Protein) - 基因编码的蛋白质产物。例如：胰岛素, 血红蛋白 (当需要与基因名区分时使用)",
            "化合物 (Compound) - 研究中涉及的小分子化合物。例如：葡萄糖, 紫杉醇",
            "药物 (Drug) - 具有治疗作用的化合物或生物制剂。例如：阿司匹林, 曲妥珠单抗",
            "实验方法 (Experimental Method) - 研究中使用的实验技术或方法。例如：Western Blot, CRISPR-Cas9, RNA-seq",
            "表型 (Phenotype) - 生物体可观察到的特征或性状。例如：肥胖, 抗药性, 细胞凋亡"
        ]
        self.logger.info(f"使用 {len(entity_types)} 个实体类型进行筛选")
        return entity_types
    
    def run(self, max_tables: int = None):
        """主执行流程"""
        self.logger.info("=" * 80)
        self.logger.info("开始实体类型筛选（基于 simple_filter 结果）")
        self.logger.info("=" * 80)
        sys.stdout.flush()
        
        # 1. 加载 simple_filter_results.json
        simple_results = self._load_simple_results()
        if not simple_results:
            self.logger.error("无法加载 simple_filter_results.json")
            sys.stdout.flush()
            return
        
        # 2. 筛选出已通过前两个条件的表格
        passed_tables = {
            table_id: result 
            for table_id, result in simple_results.items()
            if result.get("passed", False) and 
               result.get("is_real_table", False) and 
               result.get("is_data_table", False)
        }
        
        self.statistics["input_tables"] = len(passed_tables)
        self.logger.info(f"找到 {len(passed_tables)} 个已通过前两个条件的表格")
        sys.stdout.flush()
        
        # 3. 加载已有结果（支持断点续传）
        self._load_existing_results()
        
        # 4. 过滤掉已处理的表格
        already_processed = set(self.entity_results.keys())
        tables_to_process = {
            tid: result for tid, result in passed_tables.items()
            if tid not in already_processed
        }
        
        if already_processed:
            self.logger.info(f"已处理 {len(already_processed)} 个表格，剩余 {len(tables_to_process)} 个")
            sys.stdout.flush()
        
        # 限制处理数量（测试模式）
        if max_tables is not None and max_tables > 0:
            tables_to_process = dict(list(tables_to_process.items())[:max_tables])
            self.logger.info(f"[TEST MODE] 限制处理 {max_tables} 个表格")
            sys.stdout.flush()
        
        # 5. 处理表格
        if tables_to_process:
            self._process_tables(tables_to_process)
        
        # 6. 合并结果
        self._merge_results(simple_results)
        
        # 7. 计算统计信息
        self._calculate_statistics()
        
        # 8. 保存结果
        self._save_results()
        
        self.logger.info("=" * 80)
        self.logger.info("实体类型筛选完成！")
        self.logger.info("=" * 80)
        sys.stdout.flush()
    
    def _load_simple_results(self) -> Dict:
        """加载 simple_filter_results.json"""
        simple_results_file = self.output_dir / "simple_filter_results.json"
        if not simple_results_file.exists():
            self.logger.error(f"文件不存在: {simple_results_file}")
            return {}
        
        try:
            results = load_json(str(simple_results_file))
            self.logger.info(f"成功加载 {len(results)} 个表格的简单筛选结果")
            return results
        except Exception as e:
            self.logger.error(f"加载失败: {e}")
            return {}
    
    def _process_tables(self, tables_to_process: Dict):
        """处理表格（并发，并发数7）"""
        total = len(tables_to_process)
        max_workers = 7  # 并发数设置为7
        
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        
        self.logger.info(f"开始处理 {total} 个表格，使用 {max_workers} 个并发线程")
        self.logger.info("=" * 80)
        sys.stdout.flush()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_table = {
                executor.submit(self._process_single_table, table_id, result): table_id
                for table_id, result in tables_to_process.items()
            }
            
            completed = 0
            for future in as_completed(future_to_table):
                completed += 1
                table_id = future_to_table[future]
                
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"处理 {table_id} 时出错: {e}")
                    sys.stdout.flush()
                
                # 定期输出进度和保存
                with self.processed_lock:
                    if completed % 10 == 0 or completed == total:  # 每10个或最后一个输出一次
                        self._print_progress(completed, total)
                    
                    if completed % cfg.SAVE_PROGRESS_EVERY == 0:
                        self._save_intermediate_results()
                        self.logger.info(f"  [自动保存] 进度已保存到文件")
                        sys.stdout.flush()
        
        # 最终保存
        self._save_intermediate_results()
        self._print_final_progress(total)
        sys.stdout.flush()
    
    def _process_single_table(self, table_id: str, simple_result: Dict):
        """处理单个表格"""
        with self.processed_lock:
            self.processed_count += 1
        
        try:
            # 1. 加载 markdown 文件
            input_path = Path(cfg.INPUT_DIR)
            md_file = input_path / f"{table_id}.md"
            
            if not md_file.exists():
                self.logger.warning(f"文件不存在: {md_file}")
                with self.results_lock:
                    self.entity_results[table_id] = {
                        "extracted_entities": [],
                        "classification": "全都不在",
                        "matched_entities": [],
                        "unmatched_entities": [],
                        "reason": "文件不存在",
                        "status": "skipped"
                    }
                with self.stats_lock:
                    self.statistics["skipped_tables"] += 1
                return
            
            # 2. 加载并提取内容
            markdown_content = load_markdown(str(md_file))
            extracted_content = extract_relevant_sections(markdown_content)
            
            if not extracted_content or len(extracted_content.strip()) < 10:
                with self.results_lock:
                    self.entity_results[table_id] = {
                        "extracted_entities": [],
                        "classification": "全都不在",
                        "matched_entities": [],
                        "unmatched_entities": [],
                        "reason": "无表格内容",
                        "status": "skipped"
                    }
                with self.stats_lock:
                    self.statistics["skipped_tables"] += 1
                return
            
            # 3. 截断内容（统计截断信息）
            original_length = len(extracted_content)
            original_rows = extracted_content.count('\n')
            
            truncated_content = smart_truncate_for_filtering(
                extracted_content,
                sample_rows=cfg.SAMPLE_ROWS_FOR_FILTERING
            )
            
            truncated_length = len(truncated_content)
            truncated_rows = truncated_content.count('\n')
            was_truncated = truncated_length < original_length * 0.9  # 如果减少了10%以上，认为被截断
            
            if len(truncated_content) > cfg.MAX_CONTENT_LENGTH:
                truncated_content = truncated_content[:cfg.MAX_CONTENT_LENGTH] + "\n\n...(truncated)"
                was_truncated = True
            
            # 更新截断统计
            with self.stats_lock:
                if was_truncated:
                    self.statistics["truncated_tables"] += 1
                self.statistics["total_original_rows"] += original_rows
                self.statistics["total_truncated_rows"] += truncated_rows
            
            # 4. 第一步：提取实体类型（不告诉模型实体集合）
            extraction_result = self.qwen_client.extract_entities_only(
                truncated_content,
                retry_times=cfg.RETRY_TIMES,
                backoff_factor=cfg.BACKOFF_FACTOR
            )
            
            if not extraction_result:
                with self.results_lock:
                    self.entity_results[table_id] = {
                        "extracted_entities": [],
                        "classification": "全都不在",
                        "matched_entities": [],
                        "unmatched_entities": [],
                        "reason": "实体提取失败",
                        "status": "failed"
                    }
                with self.stats_lock:
                    self.statistics["error_count"] += 1
                    self.statistics["failed_extraction"] += 1
                    self.statistics["none_in"] += 1
                    self.statistics["processed_tables"] += 1
                return
            
            extracted_entities = extraction_result.get("entity_types", [])
            
            # 5. 第二步：判断提取的实体是否在既定集合中
            classification_result = self.qwen_client.classify_entities(
                extracted_entities,
                self.entity_types,
                retry_times=cfg.RETRY_TIMES,
                backoff_factor=cfg.BACKOFF_FACTOR
            )
            
            if not classification_result:
                with self.results_lock:
                    self.entity_results[table_id] = {
                        "extracted_entities": extracted_entities,
                        "classification": "全都不在",
                        "matched_entities": [],
                        "unmatched_entities": extracted_entities,
                        "reason": "实体分类失败",
                        "status": "failed"
                    }
                with self.stats_lock:
                    self.statistics["error_count"] += 1
                    self.statistics["failed_classification"] += 1
                    self.statistics["none_in"] += 1
                    self.statistics["processed_tables"] += 1
                return
            
            classification = classification_result.get("classification", "全都不在")
            matched_entities = classification_result.get("matched_entities", [])
            unmatched_entities = classification_result.get("unmatched_entities", [])
            reason = classification_result.get("reason", "")
            
            with self.results_lock:
                self.entity_results[table_id] = {
                    "extracted_entities": extracted_entities,
                    "classification": classification,
                    "matched_entities": matched_entities,
                    "unmatched_entities": unmatched_entities,
                    "reason": reason,
                    "status": "success"
                }
            
            with self.stats_lock:
                if classification == "全部在":
                    self.statistics["all_in"] += 1
                elif classification == "部分在":
                    self.statistics["partial_in"] += 1
                else:  # 全都不在
                    self.statistics["none_in"] += 1
                self.statistics["processed_tables"] += 1
            
            time.sleep(cfg.REQUEST_DELAY)
            
        except Exception as e:
            self.logger.error(f"处理 {table_id} 时出错: {e}")
            sys.stdout.flush()
            with self.results_lock:
                self.entity_results[table_id] = {
                    "extracted_entities": [],
                    "classification": "全都不在",
                    "matched_entities": [],
                    "unmatched_entities": [],
                    "reason": f"错误: {str(e)}",
                    "status": "error"
                }
            with self.stats_lock:
                self.statistics["error_count"] += 1
                self.statistics["none_in"] += 1
                self.statistics["processed_tables"] += 1
    
    def _merge_results(self, simple_results: Dict):
        """合并 simple_results 和 entity_results"""
        for table_id, simple_result in simple_results.items():
            # 如果表格未通过前两个条件，直接标记为未通过
            if not (simple_result.get("is_real_table", False) and 
                    simple_result.get("is_data_table", False)):
                self.final_results[table_id] = {
                    **simple_result,
                    "extracted_entities": [],
                    "classification": "全都不在",
                    "matched_entities": [],
                    "unmatched_entities": [],
                    "passed_entity_filter": False
                }
                continue
            
            # 如果通过了前两个条件，检查实体类型结果
            entity_result = self.entity_results.get(table_id, {})
            classification = entity_result.get("classification", "全都不在")
            
            # 只有"全部在"或"部分在"才算通过实体筛选
            passed_entity_filter = classification in ["全部在", "部分在"]
            
            self.final_results[table_id] = {
                "is_real_table": simple_result.get("is_real_table", False),
                "is_data_table": simple_result.get("is_data_table", False),
                "extracted_entities": entity_result.get("extracted_entities", []),
                "classification": classification,
                "matched_entities": entity_result.get("matched_entities", []),
                "unmatched_entities": entity_result.get("unmatched_entities", []),
                "reason": entity_result.get("reason", simple_result.get("reason", "")),
                "passed_entity_filter": passed_entity_filter,
                "status": entity_result.get("status", simple_result.get("status", "unknown"))
            }
    
    def _print_progress(self, completed: int, total: int):
        """打印实时进度信息"""
        with self.stats_lock:
            stats = dict(self.statistics)
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if completed > 0:
            progress_pct = (completed / total) * 100
            speed = completed / elapsed_time if elapsed_time > 0 else 0
            remaining = total - completed
            eta_seconds = remaining / speed if speed > 0 else 0
            eta_minutes = eta_seconds / 60
            
            elapsed_minutes = elapsed_time / 60
            
            self.logger.info("")
            self.logger.info("-" * 80)
            self.logger.info(f"📊 进度统计 [{completed}/{total}] ({progress_pct:.1f}%)")
            self.logger.info(f"⏱️  已用时间: {elapsed_minutes:.1f} 分钟 | 处理速度: {speed:.2f} 表格/秒 | 预计剩余: {eta_minutes:.1f} 分钟")
            self.logger.info(f"📈 分类统计:")
            self.logger.info(f"   - 全部在: {stats['all_in']} | 部分在: {stats['partial_in']} | 全都不在: {stats['none_in']}")
            self.logger.info(f"   - 错误数: {stats['error_count']} | 跳过: {stats['skipped_tables']}")
            self.logger.info(f"   - 提取失败: {stats['failed_extraction']} | 分类失败: {stats['failed_classification']}")
            
            if stats['truncated_tables'] > 0:
                truncate_rate = (stats['truncated_tables'] / completed) * 100 if completed > 0 else 0
                avg_original = stats['total_original_rows'] / completed if completed > 0 else 0
                avg_truncated = stats['total_truncated_rows'] / completed if completed > 0 else 0
                self.logger.info(f"✂️  截断统计: {stats['truncated_tables']} 个表格被截断 ({truncate_rate:.1f}%)")
                self.logger.info(f"   - 平均原始行数: {avg_original:.1f} | 平均截断后行数: {avg_truncated:.1f}")
            
            self.logger.info("-" * 80)
            sys.stdout.flush()
    
    def _print_final_progress(self, total: int):
        """打印最终进度信息"""
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"✅ 处理完成！总耗时: {elapsed_minutes:.1f} 分钟 ({elapsed_time:.0f} 秒)")
        self.logger.info("=" * 80)
        sys.stdout.flush()
    
    def _calculate_statistics(self):
        """计算最终统计信息"""
        total_processed = self.statistics["processed_tables"]
        
        if total_processed > 0:
            self.statistics["all_in_rate"] = self.statistics["all_in"] / total_processed
            self.statistics["partial_in_rate"] = self.statistics["partial_in"] / total_processed
            self.statistics["none_in_rate"] = self.statistics["none_in"] / total_processed
        else:
            self.statistics["all_in_rate"] = 0.0
            self.statistics["partial_in_rate"] = 0.0
            self.statistics["none_in_rate"] = 0.0
        
        # 统计通过实体筛选的表格数
        passed_count = sum(1 for r in self.final_results.values() 
                          if r.get("passed_entity_filter", False))
        self.statistics["passed_entity_filter"] = passed_count
        
        # 计算截断统计
        if total_processed > 0:
            self.statistics["truncate_rate"] = (self.statistics["truncated_tables"] / total_processed) * 100
            self.statistics["avg_original_rows"] = self.statistics["total_original_rows"] / total_processed
            self.statistics["avg_truncated_rows"] = self.statistics["total_truncated_rows"] / total_processed
        else:
            self.statistics["truncate_rate"] = 0.0
            self.statistics["avg_original_rows"] = 0.0
            self.statistics["avg_truncated_rows"] = 0.0
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 最终统计信息")
        self.logger.info("=" * 80)
        self.logger.info(f"📥 输入表格数（已通过前两个条件）: {self.statistics['input_tables']}")
        self.logger.info(f"✅ 处理表格数: {self.statistics['processed_tables']}")
        self.logger.info("")
        self.logger.info("📈 分类统计:")
        self.logger.info(f"   - 全部在: {self.statistics['all_in']} ({self.statistics['all_in_rate']:.2%})")
        self.logger.info(f"   - 部分在: {self.statistics['partial_in']} ({self.statistics['partial_in_rate']:.2%})")
        self.logger.info(f"   - 全都不在: {self.statistics['none_in']} ({self.statistics['none_in_rate']:.2%})")
        self.logger.info(f"   - 通过实体筛选: {self.statistics['passed_entity_filter']}")
        self.logger.info("")
        self.logger.info("⚠️  错误统计:")
        self.logger.info(f"   - 总错误数: {self.statistics['error_count']}")
        self.logger.info(f"   - 实体提取失败: {self.statistics['failed_extraction']}")
        self.logger.info(f"   - 实体分类失败: {self.statistics['failed_classification']}")
        self.logger.info(f"   - 跳过表格（文件不存在/无内容）: {self.statistics['skipped_tables']}")
        self.logger.info("")
        self.logger.info("✂️  截断统计:")
        self.logger.info(f"   - 被截断的表格数: {self.statistics['truncated_tables']} ({self.statistics['truncate_rate']:.1f}%)")
        self.logger.info(f"   - 平均原始行数: {self.statistics['avg_original_rows']:.1f}")
        self.logger.info(f"   - 平均截断后行数: {self.statistics['avg_truncated_rows']:.1f}")
        self.logger.info("=" * 80)
        sys.stdout.flush()
    
    def _save_results(self):
        """保存结果"""
        # 保存实体类型筛选结果
        save_json(
            self.entity_results,
            self.output_dir / "entity_filter_results.json"
        )
        
        # 保存最终合并结果
        save_json(
            self.final_results,
            self.output_dir / "entity_filter_final_results.json"
        )
        
        # 保存通过实体筛选的表格列表
        passed_tables = [
            table_id for table_id, result in self.final_results.items()
            if result.get("passed_entity_filter", False)
        ]
        save_json(
            {"filtered_tables": passed_tables, "count": len(passed_tables)},
            self.output_dir / "entity_filtered_tables.json"
        )
        
        # 保存统计信息
        save_json(
            dict(self.statistics),
            self.output_dir / "entity_filter_statistics.json"
        )
        
        self.logger.info(f"\n结果已保存到 {self.output_dir}")
        self.logger.info(f"  - entity_filter_results.json: 实体类型筛选结果")
        self.logger.info(f"  - entity_filter_final_results.json: 最终合并结果")
        self.logger.info(f"  - entity_filtered_tables.json: {len(passed_tables)} 个通过实体筛选的表格")
        self.logger.info(f"  - entity_filter_statistics.json: 统计信息")
        sys.stdout.flush()
    
    def _load_existing_results(self):
        """加载已有结果（支持断点续传）"""
        entity_results_file = self.output_dir / "entity_filter_results.json"
        if entity_results_file.exists():
            try:
                self.entity_results = load_json(str(entity_results_file))
                self.logger.info(f"加载了 {len(self.entity_results)} 个已有结果")
                sys.stdout.flush()
            except Exception as e:
                self.logger.warning(f"加载已有结果失败: {e}")
                self.entity_results = {}
    
    def _save_intermediate_results(self):
        """保存中间结果"""
        with self.results_lock:
            results_copy = dict(self.entity_results)
        save_json(
            results_copy,
            self.output_dir / "entity_filter_results.json"
        )


def main():
    """主入口"""
    filter_system = EntityFilter()
    # Use SAMPLE_SIZE from config (None = process all)
    max_tables = getattr(cfg, 'SAMPLE_SIZE', None)
    filter_system.run(max_tables=max_tables)


if __name__ == "__main__":
    main()

