"""
示例：如何使用按数字编号组织的表格文件

这个脚本展示了几种常见的使用场景
"""
import json
from pathlib import Path


def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def example1_load_single_table():
    """示例1: 加载单个表格"""
    print("=" * 80)
    print("示例1: 加载单个表格")
    print("=" * 80)
    
    # 加载"全部在"分类中的第1个表格
    table_file = Path("data/step3/output/全部在/1.json")
    
    if table_file.exists():
        table_data = load_json(table_file)
        
        print(f"表格编号: {table_data['number']}")
        print(f"原始ID: {table_data['table_id']}")
        print(f"分类: {table_data['classification']}")
        print(f"提取的实体: {table_data['extracted_entities']}")
        print(f"\n表格内容预览:")
        print(table_data['markdown_content'][:300] + "...")
    else:
        print("文件不存在")


def example2_batch_process():
    """示例2: 批量处理某个分类的所有表格"""
    print("\n" + "=" * 80)
    print("示例2: 批量处理所有表格")
    print("=" * 80)
    
    category_dir = Path("data/step3/output/全部在")
    
    # 先读取映射文件了解有多少表格
    mapping_file = category_dir / "id_mapping.json"
    if mapping_file.exists():
        mapping_data = load_json(mapping_file)
        total_count = mapping_data['total_count']
        
        print(f"分类: {mapping_data['category']}")
        print(f"总表格数: {total_count}")
        
        # 处理前10个表格作为示例
        print(f"\n处理前10个表格:")
        for i in range(1, min(11, total_count + 1)):
            table_file = category_dir / f"{i}.json"
            if table_file.exists():
                table_data = load_json(table_file)
                print(f"  {i}. {table_data['table_id']} - 实体数: {len(table_data['extracted_entities'])}")


def example3_find_by_original_id():
    """示例3: 根据原始table_id查找对应的数字编号"""
    print("\n" + "=" * 80)
    print("示例3: 根据原始table_id查找数字编号")
    print("=" * 80)
    
    target_table_id = "table_12345"  # 要查找的表格ID
    category = "全部在"
    
    category_dir = Path(f"data/step3/output/{category}")
    mapping_file = category_dir / "id_mapping.json"
    
    if mapping_file.exists():
        mapping_data = load_json(mapping_file)
        
        # 反向查找：从table_id找到数字编号
        number = None
        for num, tid in mapping_data['mapping'].items():
            if tid == target_table_id:
                number = num
                break
        
        if number:
            print(f"找到表格 {target_table_id}")
            print(f"数字编号: {number}")
            print(f"文件路径: {category_dir}/{number}.json")
            
            # 加载该表格
            table_file = category_dir / f"{number}.json"
            table_data = load_json(table_file)
            print(f"实体类型: {table_data['extracted_entities']}")
        else:
            print(f"未找到表格 {target_table_id} 在 {category} 分类中")


def example4_statistics():
    """示例4: 统计各分类的实体分布"""
    print("\n" + "=" * 80)
    print("示例4: 统计实体分布")
    print("=" * 80)
    
    from collections import Counter
    
    category = "全部在"
    category_dir = Path(f"data/step3/output/{category}")
    
    # 读取映射文件
    mapping_file = category_dir / "id_mapping.json"
    if mapping_file.exists():
        mapping_data = load_json(mapping_file)
        total_count = mapping_data['total_count']
        
        # 统计实体类型频率
        entity_counter = Counter()
        
        print(f"分析 {category} 分类的前100个表格...")
        for i in range(1, min(101, total_count + 1)):
            table_file = category_dir / f"{i}.json"
            if table_file.exists():
                table_data = load_json(table_file)
                entity_counter.update(table_data['extracted_entities'])
        
        print(f"\n最常见的实体类型 (前10):")
        for entity, count in entity_counter.most_common(10):
            print(f"  {entity}: {count}")


def example5_export_to_markdown():
    """示例5: 导出某个分类的所有表格为单独的markdown文件"""
    print("\n" + "=" * 80)
    print("示例5: 导出为Markdown文件")
    print("=" * 80)
    
    category = "全部在"
    category_dir = Path(f"data/step3/output/{category}")
    output_dir = Path(f"./exported_markdown_{category}")
    output_dir.mkdir(exist_ok=True)
    
    # 读取映射文件
    mapping_file = category_dir / "id_mapping.json"
    if mapping_file.exists():
        mapping_data = load_json(mapping_file)
        total_count = mapping_data['total_count']
        
        # 导出前10个作为示例
        print(f"导出前10个表格到 {output_dir}...")
        for i in range(1, min(11, total_count + 1)):
            table_file = category_dir / f"{i}.json"
            if table_file.exists():
                table_data = load_json(table_file)
                
                # 保存为markdown文件
                md_file = output_dir / f"{i}_{table_data['table_id']}.md"
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(f"# 表格 {i}\n\n")
                    f.write(f"**原始ID**: {table_data['table_id']}\n\n")
                    f.write(f"**分类**: {table_data['classification']}\n\n")
                    f.write(f"**实体类型**: {', '.join(table_data['extracted_entities'])}\n\n")
                    f.write(f"---\n\n")
                    f.write(table_data['markdown_content'])
        
        print(f"✅ 完成！导出到 {output_dir}")


def example6_iterate_all_categories():
    """示例6: 遍历所有分类"""
    print("\n" + "=" * 80)
    print("示例6: 遍历所有分类")
    print("=" * 80)
    
    base_dir = Path("data/step3/output")
    categories = ["全部在", "部分在", "全都不在"]
    
    for category in categories:
        category_dir = base_dir / category
        if not category_dir.exists():
            continue
        
        mapping_file = category_dir / "id_mapping.json"
        if mapping_file.exists():
            mapping_data = load_json(mapping_file)
            print(f"\n【{category}】")
            print(f"  总表格数: {mapping_data['total_count']}")
            
            # 显示前3个表格的信息
            for i in range(1, min(4, mapping_data['total_count'] + 1)):
                table_file = category_dir / f"{i}.json"
                if table_file.exists():
                    table_data = load_json(table_file)
                    print(f"    {i}. {table_data['table_id']} - 实体: {len(table_data['extracted_entities'])}个")


def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("数字编号表格文件使用示例")
    print("=" * 80)
    print()
    
    # 运行各个示例
    try:
        example1_load_single_table()
    except Exception as e:
        print(f"示例1出错: {e}")
    
    try:
        example2_batch_process()
    except Exception as e:
        print(f"示例2出错: {e}")
    
    try:
        example3_find_by_original_id()
    except Exception as e:
        print(f"示例3出错: {e}")
    
    try:
        example4_statistics()
    except Exception as e:
        print(f"示例4出错: {e}")
    
    try:
        example5_export_to_markdown()
    except Exception as e:
        print(f"示例5出错: {e}")
    
    try:
        example6_iterate_all_categories()
    except Exception as e:
        print(f"示例6出错: {e}")
    
    print("\n" + "=" * 80)
    print("所有示例完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
