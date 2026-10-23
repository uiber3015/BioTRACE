#!/usr/bin/env python3
"""
RapidTable 诊断脚本
用于测试 RapidTable 的返回格式和 GPU 配置
"""

import sys
from pathlib import Path

print("=" * 70)
print("RapidTable 诊断工具")
print("=" * 70)

# 1. 检查 RapidTable 安装
print("\n[1] 检查 RapidTable 安装...")
try:
    from rapid_table import RapidTable
    print("✓ RapidTable 已安装")
except ImportError as e:
    print(f"✗ RapidTable 未安装: {e}")
    sys.exit(1)

# 2. 检查 ONNX Runtime 配置
print("\n[2] 检查 ONNX Runtime...")
try:
    import onnxruntime as ort
    print(f"✓ ONNX Runtime 版本: {ort.__version__}")
    print(f"✓ 可用的 Execution Providers: {ort.get_available_providers()}")
    
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("✓ GPU (CUDA) 支持可用")
    else:
        print("⚠ GPU (CUDA) 支持不可用，将使用 CPU")
        print("\n  要启用 GPU，请执行以下步骤：")
        print("  1. 卸载现有 onnxruntime:")
        print("     pip uninstall onnxruntime onnxruntime-gpu -y")
        print("  2. 安装 onnxruntime-gpu (需匹配 CUDA 版本):")
        print("     pip install onnxruntime-gpu")
        print("  3. 验证 CUDA 和 cuDNN 版本兼容性:")
        print("     https://onnxruntime.ai/docs/execution-providers/CUDA-EP.html")
except ImportError as e:
    print(f"✗ ONNX Runtime 未安装: {e}")

# 3. 初始化 RapidTable
print("\n[3] 初始化 RapidTable...")
try:
    # 检查 RapidTable 的初始化方式
    import inspect
    print(f"   RapidTable 签名: {inspect.signature(RapidTable.__init__)}")
    
    # 根据签名，需要 RapidTableInput 对象
    from rapid_table import RapidTableInput
    
    # 检查 RapidTableInput 的签名和可用参数
    print(f"   RapidTableInput 签名: {inspect.signature(RapidTableInput.__init__)}")
    
    # 创建配置对象 - CPU 模式
    print("   正在创建 RapidTableInput (CPU 模式)...")
    config = RapidTableInput(
        model_type='slanet_plus',
        use_cuda=False,
        device='cpu'
    )
    print(f"   ✓ 配置创建成功: model_type={config.model_type}, use_cuda={config.use_cuda}, device={config.device}")
    
    # 初始化 RapidTable
    print("   正在初始化 RapidTable...")
    table_engine = RapidTable(config)
    print("✓ RapidTable 初始化成功！")
    
except Exception as e:
    print(f"✗ RapidTable 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试表格识别（如果提供了测试图片）
print("\n[4] 测试表格识别...")

# 查找测试图片
test_image_paths = [
    Path("step1/output/table_images/table_1.jpg"),
    Path("step1/output/table_images/table_10008.jpg"),
    Path("./step1/output/table_images/table_1.jpg"),
    Path("./step1/output/table_images/table_10008.jpg"),
]

test_image = None
for path in test_image_paths:
    if path.exists():
        test_image = path
        break

if test_image:
    print(f"   使用测试图片: {test_image}")
    try:
        result = table_engine(str(test_image))
        
        print(f"\n📊 RapidTable 返回结果分析:")
        print(f"   - 返回类型: {type(result)}")
        print(f"   - 返回内容类型名: {type(result).__name__}")
        
        # 检查不同的访问方式
        print(f"\n🔍 检查返回数据结构:")
        
        # 方式1: 对象属性
        if hasattr(result, '__dict__'):
            print(f"   - 对象属性: {list(result.__dict__.keys())}")
            if hasattr(result, 'html'):
                print(f"   - result.html 存在: {bool(result.html)}")
                if result.html:
                    print(f"   - HTML 长度: {len(result.html)} 字符")
                    print(f"   - HTML 预览: {result.html[:200]}...")
        
        # 方式2: 字典
        if isinstance(result, dict):
            print(f"   - 字典键: {list(result.keys())}")
            if 'html' in result:
                print(f"   - result['html'] 存在: {bool(result['html'])}")
                if result['html']:
                    print(f"   - HTML 长度: {len(result['html'])} 字符")
        
        # 方式3: 元组
        if isinstance(result, tuple):
            print(f"   - 元组长度: {len(result)}")
            for i, item in enumerate(result):
                print(f"   - result[{i}] 类型: {type(item).__name__}")
                if isinstance(item, str) and item.startswith('<'):
                    print(f"   - result[{i}] 是 HTML 字符串，长度: {len(item)}")
        
        # 直接打印（截断）
        result_str = str(result)
        if len(result_str) > 500:
            result_str = result_str[:500] + "..."
        print(f"\n📄 完整返回值 (截断):")
        print(f"   {result_str}")
        
        print("\n✓ 表格识别测试完成")
        
    except Exception as e:
        print(f"✗ 表格识别失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ 未找到测试图片，跳过识别测试")
    print("   期望路径: step1/output/table_images/table_1.jpg")

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)

