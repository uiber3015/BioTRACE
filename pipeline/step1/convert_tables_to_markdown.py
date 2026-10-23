#!/usr/bin/env python3
"""
步骤1B: 使用RapidTable将表格图片转换为Markdown
功能：
1. 读取table_images和table_captions
2. 使用RapidTable识别表格内容
3. 生成完整的Markdown文件（包含caption、识别结果、引用段落）
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TableToMarkdownConverter:
    """表格转Markdown转换器（依赖RapidTable）"""
    
    def __init__(
        self,
        input_root: str,
        use_cuda: bool = False,
        gpu_id: int = 0
    ):
        self.input_root = Path(input_root)
        self.images_dir = self.input_root / "table_images"
        self.captions_dir = self.input_root / "table_captions"
        self.markdown_dir = self.input_root / "table_markdown_1"
        
        # 检查输入目录
        if not self.images_dir.exists():
            raise FileNotFoundError(f"未找到图片目录: {self.images_dir}")
        if not self.captions_dir.exists():
            raise FileNotFoundError(f"未找到caption目录: {self.captions_dir}")
        
        # 创建输出目录
        self.markdown_dir.mkdir(exist_ok=True)
        
        # 统计信息
        self.stats = {
            "total_tables": 0,
            "success_count": 0,
            "failed_count": 0
        }
        
        # 初始化RapidTable
        self.table_engine = None
        self._init_rapid_table(use_cuda, gpu_id)
    
    def _init_rapid_table(self, use_cuda: bool, gpu_id: int):
        """初始化RapidTable（兼容多个版本）"""
        try:
            from rapid_table import RapidTable
            
            logger.info("正在初始化RapidTable...")
            
            # 尝试多种初始化方式以兼容不同版本
            init_success = False
            
            # 方法1: 使用 RapidTableInput (适用于 1.0.3+)
            try:
                from rapid_table import RapidTableInput
                
                if use_cuda:
                    device = f'cuda:{gpu_id}'
                    input_args = RapidTableInput(
                        model_type='slanet_plus',
                        use_cuda=True,
                        device=device
                    )
                    logger.info(f"使用GPU模式 (设备: {device})")
                else:
                    input_args = RapidTableInput(
                        model_type='slanet_plus',
                        use_cuda=False,
                        device='cpu'
                    )
                    logger.info("使用CPU模式")
                
                self.table_engine = RapidTable(input_args)
                init_success = True
                logger.info("✓ RapidTable初始化成功 (使用 RapidTableInput)")
                
            except (ImportError, TypeError) as e:
                logger.debug(f"RapidTableInput 方式失败: {e}")
            
            # 方法2: 使用字典 config (某些版本)
            if not init_success:
                try:
                    config = {
                        'model_type': 'slanet_plus',
                        'use_cuda': use_cuda,
                        'device': f'cuda:{gpu_id}' if use_cuda else 'cpu'
                    }
                    self.table_engine = RapidTable(config=config)
                    init_success = True
                    logger.info("✓ RapidTable初始化成功 (使用 config 字典)")
                except TypeError as e:
                    logger.debug(f"config 字典方式失败: {e}")
            
            # 方法3: 尝试无参数初始化
            if not init_success:
                try:
                    self.table_engine = RapidTable()
                    init_success = True
                    logger.info("✓ RapidTable初始化成功 (无参数)")
                    if use_cuda:
                        logger.warning("⚠ 无参数初始化可能不支持GPU加速")
                except TypeError as e:
                    logger.debug(f"无参数方式失败: {e}")
            
            if not init_success:
                raise RuntimeError("所有初始化方法均失败，请检查 rapid_table 版本")
            
        except ImportError as e:
            logger.error("未安装rapid_table库！")
            logger.error("请运行: pip install rapid_table")
            raise ImportError("需要安装rapid_table库") from e
        except Exception as e:
            logger.error(f"RapidTable初始化失败: {e}")
            logger.error("请尝试运行诊断脚本: python step1/diagnose_rapidtable.py")
            raise
    
    def load_caption_data(self, table_id: int) -> Optional[Dict]:
        """加载caption数据"""
        caption_file = self.captions_dir / f"table_{table_id}.json"
        
        if not caption_file.exists():
            logger.warning(f"未找到caption文件: {caption_file}")
            return None
        
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取caption失败 {caption_file}: {e}")
            return None
    
    def convert_to_markdown(self, img_path: Path, caption_data: Dict) -> str:
        """使用RapidTable将表格图片转换为Markdown"""
        try:
            # 调用RapidTable
            logger.debug(f"正在识别表格: {img_path.name}")
            result = self.table_engine(str(img_path))
            
            # 🔍 诊断：打印RapidTable返回结果类型和内容
            logger.debug(f"RapidTable返回类型: {type(result)}")
            logger.debug(f"RapidTable返回内容: {result}")
            
            # 构建Markdown内容
            markdown = ""
            
            # 添加标题
            markdown += f"# Table {caption_data['table_id']}\n\n"
            
            # 添加元信息
            markdown += f"**Source:** {caption_data['doc_id']}\n"
            if caption_data.get('page_idx') is not None:
                markdown += f"**Page:** {caption_data['page_idx']}\n"
            markdown += f"**Table Identifier:** {caption_data.get('table_identifier', 'N/A')}\n"
            markdown += "\n"
            
            # 添加caption
            if caption_data.get('table_caption'):
                markdown += "## Caption\n\n"
                for cap in caption_data['table_caption']:
                    markdown += f"{cap}\n\n"
            
            # 添加RapidTable识别结果
            markdown += "## Table Content\n\n"
            
            # 检查多种可能的返回格式
            table_html = None
            
            # 情况1: result是对象，有pred_html属性（RapidTable的实际返回格式）
            if hasattr(result, 'pred_html') and result.pred_html:
                table_html = result.pred_html
                logger.debug(f"✓ 找到HTML (通过result.pred_html)")
            # 情况2: result是对象，有html属性
            elif hasattr(result, 'html') and result.html:
                table_html = result.html
                logger.debug(f"✓ 找到HTML (通过result.html)")
            # 情况3: result是字典，有pred_html键
            elif isinstance(result, dict) and 'pred_html' in result and result['pred_html']:
                table_html = result['pred_html']
                logger.debug(f"✓ 找到HTML (通过dict['pred_html'])")
            # 情况4: result是字典，有html键
            elif isinstance(result, dict) and 'html' in result and result['html']:
                table_html = result['html']
                logger.debug(f"✓ 找到HTML (通过dict['html'])")
            # 情况5: result是元组 (html, elapse)
            elif isinstance(result, tuple) and len(result) >= 2:
                table_html = result[0]
                logger.debug(f"✓ 找到HTML (通过tuple[0])")
            
            if table_html:
                markdown += self._html_to_markdown_table(table_html)
            else:
                # 记录详细的失败信息
                logger.warning(f"❌ 表格识别失败 (table_{caption_data['table_id']})")
                logger.warning(f"   - 返回类型: {type(result)}")
                if hasattr(result, '__dict__'):
                    logger.warning(f"   - 属性: {result.__dict__}")
                elif isinstance(result, dict):
                    logger.warning(f"   - 字典键: {result.keys()}")
                markdown += "*(表格识别失败 - RapidTable未返回有效的HTML结果)*\n"
            
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
            markdown += f"*Generated by RapidTable*\n"
            
            return markdown
            
        except Exception as e:
            logger.error(f"RapidTable转换错误: {e}")
            return f"# Table {caption_data['table_id']}\n\n**Error:** {str(e)}\n"
    
    def _html_to_markdown_table(self, html_str: str) -> str:
        """简单的HTML表格转Markdown（使用html2text库）"""
        try:
            # 尝试使用html2text
            import html2text
            h = html2text.HTML2Text()
            h.body_width = 0  # 不换行
            h.ignore_images = True
            h.ignore_links = False
            return h.handle(html_str)
        except ImportError:
            # 如果没有html2text，返回原HTML
            logger.warning("未安装html2text库，将返回原始HTML")
            logger.warning("建议运行: pip install html2text")
            return f"```html\n{html_str}\n```\n"
    
    def process_single_table(self, table_id: int) -> bool:
        """处理单个表格"""
        try:
            # 加载caption数据
            caption_data = self.load_caption_data(table_id)
            if caption_data is None:
                return False
            
            # 获取图片路径
            img_name = caption_data.get('image_file')
            if not img_name:
                logger.warning(f"Table {table_id}: 未找到image_file字段")
                return False
            
            img_path = self.images_dir / img_name
            if not img_path.exists():
                logger.warning(f"Table {table_id}: 图片不存在 {img_path}")
                return False
            
            # 转换为Markdown
            markdown_content = self.convert_to_markdown(img_path, caption_data)
            
            # 保存Markdown文件
            markdown_file = self.markdown_dir / f"table_{table_id}.md"
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"✓ Table {table_id}: 转换成功")
            return True
            
        except Exception as e:
            logger.error(f"✗ Table {table_id}: 转换失败 - {e}")
            return False
    
    def run(self):
        """运行完整流程"""
        logger.info("=" * 60)
        logger.info("开始将表格图片转换为Markdown")
        logger.info("=" * 60)
        
        # 获取所有caption文件
        caption_files = sorted(self.captions_dir.glob("table_*.json"))
        self.stats["total_tables"] = len(caption_files)
        
        if not caption_files:
            logger.warning("未找到任何caption文件！")
            return
        
        logger.info(f"找到 {len(caption_files)} 个表格")
        
        # 逐个处理
        for caption_file in tqdm(caption_files, desc="转换表格"):
            # 从文件名提取table_id
            table_id = int(caption_file.stem.split('_')[1])
            
            if self.process_single_table(table_id):
                self.stats["success_count"] += 1
            else:
                self.stats["failed_count"] += 1
        
        # 生成汇总报告
        self.generate_summary_report()
        
        logger.info("=" * 60)
        logger.info("转换完成！")
        logger.info("=" * 60)
    
    def generate_summary_report(self):
        """生成汇总报告"""
        report = {
            "statistics": {
                "total_tables": self.stats["total_tables"],
                "success_count": self.stats["success_count"],
                "failed_count": self.stats["failed_count"]
            },
            "output_directory": str(self.markdown_dir),
            "markdown_files": f"{len(list(self.markdown_dir.glob('*.md')))} files"
        }
        
        report_path = self.input_root / "markdown_conversion_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info("\n" + "=" * 60)
        logger.info("转换汇总")
        logger.info("=" * 60)
        logger.info(f"表格总数: {self.stats['total_tables']}")
        logger.info(f"成功转换: {self.stats['success_count']}")
        logger.info(f"失败数量: {self.stats['failed_count']}")
        logger.info(f"成功率: {self.stats['success_count']/self.stats['total_tables']*100:.1f}%")
        logger.info(f"报告保存至: {report_path}")
        logger.info(f"Markdown文件目录: {self.markdown_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="步骤1B: 使用RapidTable将表格图片转换为Markdown"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="./step1/output",
        help="输入根目录（包含table_images和table_captions的目录）"
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="使用GPU加速"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID（当use-cuda时有效）"
    )
    
    args = parser.parse_args()
    
    converter = TableToMarkdownConverter(
        input_root=args.input_root,
        use_cuda=args.use_cuda,
        gpu_id=args.gpu_id
    )
    
    converter.run()


if __name__ == "__main__":
    main()

