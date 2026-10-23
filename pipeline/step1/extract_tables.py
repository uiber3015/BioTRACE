#!/usr/bin/env python3
"""
步骤1: 从PDF解析结果中提取表格图片并转换为Markdown
功能：
1. 解压ZIP文件
2. 从content_list_process.json中提取type="table"的表格
3. 复制表格图片到新文件夹
4. 保存表格caption信息
5. 使用RapidTable将表格图片转换为Markdown
"""

import os
import json
import zipfile
import shutil
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import hashlib
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TableExtractor:
    """表格提取器"""
    
    def __init__(
        self,
        input_root: str,
        output_root: str,
        extract_zip: bool = True,
        use_rapid_table: bool = True
    ):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.should_extract_zip = extract_zip
        self.use_rapid_table = use_rapid_table
        
        # 创建输出目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_root / "table_images"
        self.captions_dir = self.output_root / "table_captions"
        self.markdown_dir = self.output_root / "table_markdown"
        
        self.images_dir.mkdir(exist_ok=True)
        self.captions_dir.mkdir(exist_ok=True)
        self.markdown_dir.mkdir(exist_ok=True)
        
        # 统计信息
        self.stats = {
            "total_zips": 0,
            "processed_docs": 0,
            "failed_docs": 0,
            "total_tables": 0,
            "table_counter": 0  # 全局表格编号
        }
        
        # 初始化RapidTable
        self.table_engine = None
        if self.use_rapid_table:
            self._init_rapid_table()
    
    def _init_rapid_table(self):
        """初始化RapidTable"""
        try:
            from rapid_table import RapidTable, RapidTableInput, ModelType
            
            logger.info("正在初始化RapidTable...")
            input_args = RapidTableInput(
                model_type=ModelType.UNITABLE,
                engine_cfg={"use_cuda": False}  # 如果有GPU，设置为True
            )
            self.table_engine = RapidTable(input_args)
            logger.info("RapidTable初始化成功！")
        except ImportError:
            logger.warning("未安装rapid_table，将跳过Markdown转换。请运行: pip install rapid_table")
            self.use_rapid_table = False
        except Exception as e:
            logger.error(f"RapidTable初始化失败: {e}")
            self.use_rapid_table = False
    
    def scan_zip_files(self) -> List[Path]:
        """扫描所有ZIP文件"""
        logger.info(f"正在扫描 {self.input_root} 中的ZIP文件...")
        
        zip_files = []
        if self.input_root.is_file() and self.input_root.suffix == '.zip':
            # 如果直接指定了ZIP文件
            zip_files.append(self.input_root)
        else:
            # 直接递归查找所有ZIP文件（更高效）
            zip_files = sorted(self.input_root.rglob("*.zip"))
        
        self.stats["total_zips"] = len(zip_files)
        logger.info(f"找到 {len(zip_files)} 个ZIP文件")
        return zip_files
    
    def extract_zip(self, zip_path: Path) -> Optional[Path]:
        """解压ZIP文件"""
        extract_dir = zip_path.parent / zip_path.stem
        
        # 如果已解压且包含必要文件，跳过
        if extract_dir.exists():
            if (extract_dir / "content_list_process.json").exists():
                logger.debug(f"已解压: {zip_path.name}")
                return extract_dir
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"解压完成: {zip_path.name}")
            return extract_dir
        except Exception as e:
            logger.error(f"解压失败 {zip_path}: {e}")
            return None
    
    def _extract_table_identifier(self, caption_list: List[str], default_num: int) -> str:
        """从caption中提取表格标识符（如 'Table 1', 'Table S1' 等）"""
        if not caption_list:
            return f"Table {default_num}"
        
        # 合并所有caption文本
        caption_text = " ".join(caption_list)
        
        # 尝试匹配常见的表格标识模式
        patterns = [
            r'(Table\s+S?\d+[A-Z]?)',  # Table 1, Table S1, Table 1A
            r'(TABLE\s+\d+)',           # TABLE 1
            r'(Tab\.\s*\d+)',           # Tab. 1
            r'(表\s*\d+)',              # 中文：表 1
        ]
        
        for pattern in patterns:
            match = re.search(pattern, caption_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # 如果没有匹配，返回默认值
        return f"Table {default_num}"
    
    def _find_table_mentions(
        self, 
        table_id: str, 
        caption_list: List[str], 
        all_paragraphs: List[str]
    ) -> List[str]:
        """在段落中查找提及该表格的段落"""
        if not all_paragraphs:
            return []
        
        mentioned = []
        try:
            # 创建多个搜索模式
            search_patterns = [
                r'(?<!\w)' + re.escape(table_id) + r'(?!\w)',  # 精确匹配表格ID
            ]
            
            # 如果caption包含更具体的描述，也可以搜索
            # 但要小心避免过度匹配
            
            for paragraph in all_paragraphs:
                if not paragraph:
                    continue
                    
                # 检查是否有任何模式匹配
                for pattern in search_patterns:
                    try:
                        matches = re.findall(pattern, paragraph, flags=re.IGNORECASE)
                        if matches:
                            mentioned.append(paragraph)
                            break  # 找到一个匹配就够了，避免重复添加
                    except re.error as e:
                        logger.warning(f"正则表达式错误: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"查找表格引用时出错 '{table_id}': {str(e)}")
        
        return mentioned
    
    def extract_tables_from_doc(self, doc_dir: Path) -> List[Dict[str, Any]]:
        """从单个文档中提取表格信息"""
        content_file = doc_dir / "content_list_process.json"
        article_file = doc_dir / "article.json"
        
        if not content_file.exists():
            logger.warning(f"未找到 content_list_process.json: {doc_dir.name}")
            return []
        
        try:
            with open(content_file, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
        except Exception as e:
            logger.error(f"读取JSON失败 {content_file}: {e}")
            return []
        
        # 读取article.json以获取段落信息用于查找引用
        all_paragraphs = []
        if article_file.exists():
            try:
                with open(article_file, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    # 提取所有body_div类型的段落
                    all_paragraphs = [
                        p.get('paragraph', '') 
                        for p in article_data.get('paragraphs', [])
                        if p.get('type') == 'body_div' and p.get('paragraph')
                    ]
            except Exception as e:
                logger.warning(f"读取article.json失败 {article_file}: {e}")
        
        tables = []
        table_counter = 1  # 用于为没有明确标识的表格编号
        
        for item in content_list:
            if item.get("type") == "table":
                # 提取表格标识（用于搜索引用）
                table_caption = item.get("table_caption", [])
                table_id = self._extract_table_identifier(table_caption, table_counter)
                
                # 搜索提及该表格的段落
                mentioned_paragraphs = self._find_table_mentions(
                    table_id, 
                    table_caption, 
                    all_paragraphs
                )
                
                tables.append({
                    "doc_id": doc_dir.name,
                    "doc_dir": doc_dir,
                    "img_path": item.get("img_path", ""),
                    "table_caption": table_caption,
                    "table_footnote": item.get("table_footnote", []),
                    "table_body": item.get("table_body", ""),
                    "page_idx": item.get("page_idx", None),
                    "poly": item.get("poly", []),
                    "table_identifier": table_id,
                    "mentioned_paragraphs": mentioned_paragraphs
                })
                table_counter += 1
        
        return tables
    
    def process_table(self, table_info: Dict[str, Any]) -> bool:
        """处理单个表格：复制图片、保存caption、转换为Markdown"""
        try:
            # 增加全局计数器
            self.stats["table_counter"] += 1
            table_id = self.stats["table_counter"]
            
            doc_id = table_info["doc_id"]
            img_rel_path = table_info["img_path"]
            
            # 源图片路径
            src_img_path = table_info["doc_dir"] / img_rel_path
            if not src_img_path.exists():
                logger.warning(f"图片不存在: {src_img_path}")
                return False
            
            # 目标文件名：table_1.jpg, table_2.jpg, ...
            img_extension = src_img_path.suffix
            dst_img_name = f"table_{table_id}{img_extension}"
            dst_img_path = self.images_dir / dst_img_name
            
            # 复制图片
            shutil.copy2(src_img_path, dst_img_path)
            logger.debug(f"复制图片: {dst_img_name}")
            
            # 保存caption信息（包含引用段落）
            caption_data = {
                "table_id": table_id,
                "image_file": dst_img_name,
                "doc_id": doc_id,
                "page_idx": table_info.get("page_idx"),
                "table_caption": table_info.get("table_caption", []),
                "table_footnote": table_info.get("table_footnote", []),
                "table_identifier": table_info.get("table_identifier", f"Table {table_id}"),
                "mentioned_paragraphs": table_info.get("mentioned_paragraphs", []),
                "mentioned_count": len(table_info.get("mentioned_paragraphs", [])),
                "source_path": str(src_img_path)
            }
            
            caption_file = self.captions_dir / f"table_{table_id}.json"
            with open(caption_file, 'w', encoding='utf-8') as f:
                json.dump(caption_data, f, ensure_ascii=False, indent=2)
            
            # 使用RapidTable转换为Markdown
            if self.use_rapid_table and self.table_engine is not None:
                try:
                    markdown_content = self._convert_to_markdown(dst_img_path, caption_data)
                    markdown_file = self.markdown_dir / f"table_{table_id}.md"
                    with open(markdown_file, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    logger.debug(f"生成Markdown: table_{table_id}.md")
                except Exception as e:
                    logger.error(f"Markdown转换失败 table_{table_id}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"处理表格失败: {e}")
            return False
    
    def _convert_to_markdown(self, img_path: Path, caption_data: Dict) -> str:
        """使用RapidTable将表格图片转换为Markdown"""
        try:
            # 调用RapidTable
            result = self.table_engine(str(img_path))
            
            # 构建Markdown内容
            markdown = ""
            
            # 添加标题
            markdown += f"# Table {caption_data['table_id']}\n\n"
            
            # 添加元信息
            markdown += f"**Source:** {caption_data['doc_id']}\n"
            if caption_data.get('page_idx') is not None:
                markdown += f"**Page:** {caption_data['page_idx']}\n"
            markdown += "\n"
            
            # 添加caption
            if caption_data.get('table_caption'):
                markdown += "## Caption\n\n"
                for cap in caption_data['table_caption']:
                    markdown += f"{cap}\n\n"
            
            # 添加RapidTable识别结果
            markdown += "## Table Content\n\n"
            
            # 如果result有HTML，转换为Markdown表格
            if hasattr(result, 'html') and result.html:
                markdown += self._html_to_markdown_table(result.html)
            elif isinstance(result, dict) and 'html' in result:
                markdown += self._html_to_markdown_table(result['html'])
            else:
                markdown += "*(表格识别失败)*\n"
            
            # 添加footnote
            if caption_data.get('table_footnote'):
                markdown += "\n## Footnotes\n\n"
                for note in caption_data['table_footnote']:
                    markdown += f"- {note}\n"
            
            # 添加引用该表格的段落
            mentioned_paragraphs = caption_data.get('mentioned_paragraphs', [])
            if mentioned_paragraphs:
                markdown += "\n## Mentioned in Text\n\n"
                markdown += f"*This table is mentioned in {len(mentioned_paragraphs)} paragraph(s):*\n\n"
                for idx, para in enumerate(mentioned_paragraphs, 1):
                    # 截断过长的段落（可选）
                    para_preview = para if len(para) <= 500 else para[:500] + "..."
                    markdown += f"### Reference {idx}\n\n"
                    markdown += f"{para_preview}\n\n"
            else:
                markdown += "\n## Mentioned in Text\n\n"
                markdown += "*No explicit mentions found in the text.*\n\n"
            
            markdown += "\n---\n"
            markdown += f"*Image file: {caption_data['image_file']}*\n"
            markdown += f"*Table identifier: {caption_data.get('table_identifier', 'N/A')}*\n"
            
            return markdown
            
        except Exception as e:
            logger.error(f"RapidTable转换错误: {e}")
            return f"# Table {caption_data['table_id']}\n\n**Error:** {str(e)}\n"
    
    def _html_to_markdown_table(self, html_str: str) -> str:
        """简单的HTML表格转Markdown（可选：使用html2text库优化）"""
        try:
            # 尝试使用html2text
            import html2text
            h = html2text.HTML2Text()
            h.body_width = 0  # 不换行
            return h.handle(html_str)
        except ImportError:
            # 如果没有html2text，返回原HTML
            return f"```html\n{html_str}\n```\n"
    
    def process_single_document(self, doc_dir: Path) -> int:
        """处理单个文档，返回提取的表格数量"""
        try:
            tables = self.extract_tables_from_doc(doc_dir)
            
            if not tables:
                logger.debug(f"未找到表格: {doc_dir.name}")
                return 0
            
            success_count = 0
            for table_info in tables:
                if self.process_table(table_info):
                    success_count += 1
            
            logger.info(f"处理 {doc_dir.name}: 提取 {success_count}/{len(tables)} 个表格")
            return success_count
            
        except Exception as e:
            logger.error(f"处理文档失败 {doc_dir.name}: {e}")
            return 0
    
    def run(self):
        """运行完整流程"""
        logger.info("=" * 60)
        logger.info("开始提取表格并转换为Markdown")
        logger.info("=" * 60)
        
        # 1. 扫描ZIP文件
        zip_files = self.scan_zip_files()
        
        if not zip_files:
            logger.warning("未找到任何ZIP文件！")
            return
        
        # 2. 逐个处理
        for zip_path in tqdm(zip_files, desc="处理ZIP文件"):
            # 解压（如果需要）
            if self.should_extract_zip:
                doc_dir = self.extract_zip(zip_path)
                if doc_dir is None:
                    self.stats["failed_docs"] += 1
                    continue
            else:
                doc_dir = zip_path.parent / zip_path.stem
                if not doc_dir.exists():
                    continue
            
            # 处理文档
            table_count = self.process_single_document(doc_dir)
            
            if table_count > 0:
                self.stats["processed_docs"] += 1
                self.stats["total_tables"] += table_count
            else:
                self.stats["failed_docs"] += 1
        
        # 3. 生成汇总报告
        self.generate_summary_report()
        
        logger.info("=" * 60)
        logger.info("处理完成！")
        logger.info("=" * 60)
    
    def generate_summary_report(self):
        """生成汇总报告"""
        report = {
            "statistics": {
                "total_zips": self.stats["total_zips"],
                "processed_docs": self.stats["processed_docs"],
                "failed_docs": self.stats["failed_docs"],
                "total_tables": self.stats["total_tables"]
            },
            "output_structure": {
                "table_images": f"{len(list(self.images_dir.glob('*')))} files",
                "table_captions": f"{len(list(self.captions_dir.glob('*.json')))} files",
                "table_markdown": f"{len(list(self.markdown_dir.glob('*.md')))} files"
            }
        }
        
        report_path = self.output_root / "extraction_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info("\n" + "=" * 60)
        logger.info("提取汇总")
        logger.info("=" * 60)
        logger.info(f"扫描的ZIP文件: {self.stats['total_zips']}")
        logger.info(f"成功处理的文档: {self.stats['processed_docs']}")
        logger.info(f"失败的文档: {self.stats['failed_docs']}")
        logger.info(f"提取的表格总数: {self.stats['total_tables']}")
        logger.info(f"报告保存至: {report_path}")
        logger.info(f"\n输出目录:")
        logger.info(f"  - 表格图片: {self.images_dir}")
        logger.info(f"  - 表格标题: {self.captions_dir}")
        logger.info(f"  - Markdown文件: {self.markdown_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="步骤1: 从PDF解析结果中提取表格图片并转换为Markdown"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/source",
        help="输入根目录（包含所有ZIP文件的目录）"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./step1/output",
        help="输出根目录"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="不解压ZIP（假设已解压）"
    )
    parser.add_argument(
        "--no-rapid-table",
        action="store_true",
        help="不使用RapidTable转换Markdown"
    )
    
    args = parser.parse_args()
    
    extractor = TableExtractor(
        input_root=args.input_root,
        output_root=args.output_root,
        extract_zip=not args.no_extract,
        use_rapid_table=not args.no_rapid_table
    )
    
    extractor.run()


if __name__ == "__main__":
    main()

