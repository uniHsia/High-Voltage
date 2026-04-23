# 测试健康评估模型
import sys
sys.path.insert(0, r'E:\hsia_file\dsx\csnetwork\compitition\dashboard')

try:
    print("开始测试 get_health_prediction_data()...")
    print("这可能需要1-2分钟，请耐心等待...")

    from main_predict_2 import get_health_prediction_data
    result = get_health_prediction_data()

    print("\n" + "=" * 80)
    print("测试成功！返回结果：")
    print("=" * 80)
    print(f"设备名称: {result['device_name']}")
    print(f"HI指数: {result['hi_score']}")
    print(f"状态: {result['status']}")
    print(f"RUL预测: {result['rul_days']} 天")
    print(f"返回字段: {list(result.keys())}")

except Exception as e:
    print("\n" + "=" * 80)
    print("测试失败！错误信息：")
    print("=" * 80)
    import traceback
    traceback.print_exc()
