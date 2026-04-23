# 测试模型调用是否正常
import sys
import os

# 添加路径
sys.path.insert(0, r'E:\hsia_file\dsx\csnetwork\compitition\dashboard')

print("=" * 80)
print("测试1: 检查模型文件是否能导入")
print("=" * 80)

try:
    from main_Diagnosis_1 import get_diagnosis_data
    print("OK: main_Diagnosis_1 导入成功")
except Exception as e:
    print(f"ERROR: main_Diagnosis_1 导入失败: {e}")

try:
    from main_predict_2 import get_health_prediction_data
    print("OK: main_predict_2 导入成功")
except Exception as e:
    print(f"ERROR: main_predict_2 导入失败: {e}")

print("\n" + "=" * 80)
print("测试2: 检查PyTorch是否可用")
print("=" * 80)

try:
    import torch
    print(f"OK: PyTorch版本: {torch.__version__}")
    print(f"OK: CUDA可用: {torch.cuda.is_available()}")
except Exception as e:
    print(f"ERROR: PyTorch导入失败: {e}")

print("\n" + "=" * 80)
print("测试3: 检查其他依赖")
print("=" * 80)

dependencies = ['numpy', 'matplotlib', 'sklearn', 'django']
for dep in dependencies:
    try:
        module = __import__(dep)
        print(f"OK: {dep} 可用")
    except Exception as e:
        print(f"ERROR: {dep} 导入失败: {e}")

print("\n" + "=" * 80)
print("测试4: 尝试调用诊断模型")
print("=" * 80)

try:
    print("正在调用 get_diagnosis_data()...")
    data = get_diagnosis_data()
    print("OK: 模型调用成功！")
    print(f"返回数据包含以下字段: {list(data.keys())}")
    print(f"诊断结果: {data['discharge_type']} (置信度: {data['confidence']}%)")
except Exception as e:
    print(f"ERROR: 模型调用失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)