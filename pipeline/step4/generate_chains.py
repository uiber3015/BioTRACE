"""Step 4 主脚本 - 知识链生成（使用本地 Qwen API）"""
import json
import time
import logging
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from prompt_template import SYSTEM_PROMPT, build_user_prompt
from validators import parse_and_validate


# 配置日志
config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_table_data(json_file):
    """加载表格数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_context(parsed_content, max_length=500):
    """提取并限制context长度"""
    mentioned = parsed_content.get('mentioned_in_text', '')
    
    if not mentioned:
        return ""
    
    # 移除标题部分，只保留实际内容
    if '### Reference 1' in mentioned:
        context = mentioned.split('### Reference 1', 1)[1].strip()
    else:
        context = mentioned.strip()
    
    # 移除前导的 "*This table is mentioned*" 部分
    if '*This table is mentioned' in context:
        lines = context.split('\n')
        # 跳过前两行（标题行）
        context = '\n'.join([l for l in lines if not l.startswith('*This table')])
    
    # 限制长度
    if len(context) > max_length:
        context = context[:max_length] + "..."
    
    return context.strip()


def call_qwen_api(context, caption, table_content, retry_times=3):
    """
    调用本地 Qwen API 生成知识链
    参照 step3/qwen_api.py 的调用方式
    """
    user_prompt = build_user_prompt(context, caption, table_content)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.API_TOKEN}"
    }
    
    data = {
        "model": config.MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_TOKENS,
        "stream": False
    }
    
    for attempt in range(retry_times):
        try:
            response = requests.post(
                config.API_URL,
                headers=headers,
                json=data,
                timeout=config.TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            return content, None
        
        except requests.Timeout:
            error_msg = "请求超时"
            logger.warning(f"API 调用超时 (尝试 {attempt+1}/{retry_times})")
            if attempt < retry_times - 1:
                time.sleep(config.RETRY_DELAY * (attempt + 1))
            else:
                return None, error_msg
        
        except requests.RequestException as e:
            error_msg = f"请求错误: {str(e)}"
            logger.warning(f"API 调用失败 (尝试 {attempt+1}/{retry_times}): {error_msg}")
            if attempt < retry_times - 1:
                time.sleep(config.RETRY_DELAY * (attempt + 1))
            else:
                return None, error_msg
        
        except (KeyError, json.JSONDecodeError) as e:
            error_msg = f"响应解析错误: {str(e)}"
            logger.error(f"API 响应格式错误: {error_msg}")
            return None, error_msg
        
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"处理异常: {error_msg}")
            return None, error_msg
    
    return None, "达到最大重试次数"


def process_single_table(table_file):
    """处理单个表格"""
    table_id = table_file.stem
    
    # 检查是否已处理
    clean_output = config.CLEAN_CHAINS_DIR / f"{table_id}.json"
    if clean_output.exists():
        return {
            'table_id': table_id,
            'status': 'skipped',
            'reason': '已存在'
        }
    
    try:
        # 1. 加载表格数据
        data = load_table_data(table_file)
        parsed = data.get('parsed_content', {})
        
        # 2. 提取字段
        context = extract_context(parsed, config.MAX_CONTEXT_LENGTH)
        caption = parsed.get('caption', '')
        table_content = parsed.get('table_content', '')
        
        if not table_content or len(table_content.strip()) == 0:
            logger.warning(f"[SKIP] {table_id} - 表格内容为空")
            return {
                'table_id': table_id,
                'status': 'skipped',
                'reason': '表格内容为空'
            }
        
        # 3. 调用API
        logger.info(f"[PROCESSING] {table_id}")
        raw_output, error = call_qwen_api(context, caption, table_content, config.RETRY_TIMES)
        
        if error:
            logger.error(f"[FAILED] {table_id} - {error}")
            return {
                'table_id': table_id,
                'status': 'failed',
                'reason': error,
                'original_file': str(table_file)
            }
        
        # 4. 保存原始输出
        raw_output_file = config.RAW_CHAINS_DIR / f"{table_id}.txt"
        raw_output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_output_file, 'w', encoding='utf-8') as f:
            f.write(raw_output)
        
        # 5. 验证和清洗
        valid_chains, validation_msg = parse_and_validate(raw_output)
        
        if valid_chains is None:
            logger.error(f"[VALIDATION_FAILED] {table_id} - {validation_msg}")
            return {
                'table_id': table_id,
                'status': 'validation_failed',
                'reason': validation_msg,
                'original_file': str(table_file),
                'raw_output_file': str(raw_output_file)
            }
        
        # 6. 保存清洗后的结果
        output_data = {
            'table_id': table_id,
            'original_table_id': data.get('table_id'),
            'number': data.get('number'),
            'source_file': str(table_file),
            'chain_count': len(valid_chains),
            'chains': valid_chains,
            'validation_message': validation_msg,
            'metadata': {
                'caption': caption,
                'context_length': len(context),
                'table_content_length': len(table_content)
            }
        }
        
        clean_output.parent.mkdir(parents=True, exist_ok=True)
        with open(clean_output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[SUCCESS] {table_id} - {len(valid_chains)} 条知识链")
        return {
            'table_id': table_id,
            'status': 'success',
            'chain_count': len(valid_chains),
            'validation_message': validation_msg
        }
    
    except Exception as e:
        logger.error(f"[ERROR] {table_id} - {str(e)}", exc_info=True)
        return {
            'table_id': table_id,
            'status': 'error',
            'reason': str(e),
            'original_file': str(table_file)
        }


def main():
    """主函数"""
    print("=" * 80)
    print("Step 4: 知识链生成")
    print("=" * 80)
    
    logger.info("=" * 80)
    logger.info("Step 4: 知识链生成")
    logger.info("=" * 80)
    
    # 创建输出目录
    config.RAW_CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    config.CLEAN_CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    config.FAILED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取输入文件列表
    input_dir = config.INPUT_BASE_DIR / config.INPUT_CATEGORY
    if not input_dir.exists():
        logger.error(f"❌ 输入目录不存在: {input_dir}")
        return
    
    table_files = sorted([
        f for f in input_dir.glob("*.json")
        if f.stem.isdigit()  # 只处理数字命名的文件
    ])
    
    if config.LIMIT_COUNT:
        table_files = table_files[:config.LIMIT_COUNT]
        logger.info(f"⚠️  限制处理数量: {config.LIMIT_COUNT}")
    
    logger.info(f"📂 输入目录: {input_dir}")
    logger.info(f"📊 找到 {len(table_files)} 个表格文件")
    logger.info(f"⚙️  并发数: {config.MAX_CONCURRENCY}")
    logger.info(f"🔗 API: {config.API_URL}")
    logger.info(f"🤖 模型: {config.MODEL_NAME}")
    
    if len(table_files) == 0:
        logger.error("❌ 没有找到可处理的表格文件")
        return
    
    # 并发处理
    results = []
    start_time = time.time()
    
    print(f"\n开始处理 {len(table_files)} 个表格...\n")
    
    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as executor:
        futures = {executor.submit(process_single_table, f): f for f in table_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            # 进度提示
            if i % 10 == 0 or result['status'] in ['failed', 'error']:
                elapsed = time.time() - start_time
                speed = i / elapsed if elapsed > 0 else 0
                remaining = (len(table_files) - i) / speed if speed > 0 else 0
                
                status_emoji = {
                    'success': '✅',
                    'skipped': '⏭️',
                    'failed': '❌',
                    'validation_failed': '⚠️',
                    'error': '💥'
                }.get(result['status'], '❓')
                
                print(f"{status_emoji} [{i}/{len(table_files)}] {result['table_id']} - {result['status']}")
                
                if i % 10 == 0:
                    logger.info(
                        f"进度: {i}/{len(table_files)} ({i/len(table_files)*100:.1f}%) - "
                        f"速度: {speed:.2f}个/秒 - 预计剩余: {remaining/60:.1f}分钟"
                    )
    
    # 统计结果
    status_count = {}
    total_chains = 0
    
    for r in results:
        status = r['status']
        status_count[status] = status_count.get(status, 0) + 1
        if status == 'success':
            total_chains += r.get('chain_count', 0)
    
    # 保存统计报告
    summary = {
        'total_tables': len(table_files),
        'processed_time_seconds': time.time() - start_time,
        'processed_time_minutes': (time.time() - start_time) / 60,
        'status_distribution': status_count,
        'total_chains_generated': total_chains,
        'average_chains_per_table': total_chains / status_count.get('success', 1),
        'config': {
            'input_category': config.INPUT_CATEGORY,
            'model': config.MODEL_NAME,
            'temperature': config.TEMPERATURE,
            'max_concurrency': config.MAX_CONCURRENCY
        },
        'results': results
    }
    
    summary_file = config.OUTPUT_BASE_DIR / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 保存失败记录
    failed_records = [r for r in results if r['status'] in ['failed', 'validation_failed', 'error']]
    if failed_records:
        failed_file = config.FAILED_DIR / "failed_log.jsonl"
        with open(failed_file, 'w', encoding='utf-8') as f:
            for record in failed_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"💾 失败记录已保存: {failed_file}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("处理完成")
    print("=" * 80)
    
    logger.info("\n" + "=" * 80)
    logger.info("处理完成")
    logger.info("=" * 80)
    logger.info(f"📊 总表格数: {len(table_files)}")
    logger.info(f"⏱️  处理时间: {summary['processed_time_minutes']:.1f} 分钟")
    logger.info(f"📈 状态分布:")
    for status, count in sorted(status_count.items()):
        percentage = count / len(table_files) * 100
        emoji = {'success': '✅', 'skipped': '⏭️', 'failed': '❌', 
                 'validation_failed': '⚠️', 'error': '💥'}.get(status, '❓')
        logger.info(f"   {emoji} {status}: {count} ({percentage:.1f}%)")
    logger.info(f"🔗 生成知识链总数: {total_chains}")
    logger.info(f"📊 平均每表: {summary['average_chains_per_table']:.2f} 条")
    logger.info(f"📄 统计报告: {summary_file}")
    logger.info("=" * 80)
    
    print(f"\n✅ 成功: {status_count.get('success', 0)}")
    print(f"⏭️  跳过: {status_count.get('skipped', 0)}")
    print(f"❌ 失败: {status_count.get('failed', 0) + status_count.get('validation_failed', 0) + status_count.get('error', 0)}")
    print(f"🔗 生成知识链: {total_chains} 条")
    print(f"\n📄 详细报告: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
