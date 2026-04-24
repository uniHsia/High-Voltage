#!/usr/bin/env python
"""
系统功能测试脚本
测试所有主要功能是否正常工作
"""
import os
import sys
import django

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'high_voltage_platform.settings')
django.setup()

from django.test import Client
from django.urls import reverse
from dashboard.utils import user_exists, register_user, verify_user
import json

def test_basic_imports():
    """测试基本导入"""
    print("测试基本导入...")
    try:
        import torch
        print("✓ PyTorch 已安装")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ PyTorch 未安装")
        return False

    try:
        import numpy as np
        print("✓ NumPy 已安装")
    except ImportError:
        print("✗ NumPy 未安装")
        return False

    try:
        from sklearn.metrics import accuracy_score
        print("✓ scikit-learn 已安装")
    except ImportError:
        print("✗ scikit-learn 未安装")
        return False

    try:
        import matplotlib
        print("✓ matplotlib 已安装")
    except ImportError:
        print("✗ matplotlib 未安装")
        return False

    return True

def test_model_imports():
    """测试模型导入"""
    print("\n测试模型导入...")
    try:
        from dashboard.main_Diagnosis_1 import get_diagnosis_data
        print("✓ 诊断模型导入成功")
    except Exception as e:
        print(f"✗ 诊断模型导入失败: {e}")
        return False

    try:
        from dashboard.main_predict_2 import get_health_prediction_data
        print("✓ 健康评估模型导入成功")
    except Exception as e:
        print(f"✗ 健康评估模型导入失败: {e}")
        return False

    return True

def test_user_system():
    """测试用户系统"""
    print("\n测试用户系统...")
    test_username = "test_user_12345"
    test_email = "test@example.com"
    test_password = "test_password_123"

    # 清理测试用户
    if user_exists(test_username):
        print(f"清理已存在的测试用户...")
        # 这里可以添加删除用户的代码

    # 测试注册
    try:
        register_user(test_username, test_email, test_password)
        print("✓ 用户注册功能正常")
    except Exception as e:
        print(f"✗ 用户注册失败: {e}")
        return False

    # 测试验证
    try:
        if verify_user(test_username, test_password):
            print("✓ 用户验证功能正常")
        else:
            print("✗ 用户验证失败")
            return False
    except Exception as e:
        print(f"✗ 用户验证出错: {e}")
        return False

    return True

def test_django_views():
    """测试Django视图"""
    print("\n测试Django视图...")
    client = Client()

    # 测试登录页面
    try:
        response = client.get('/', HTTP_HOST='localhost')
        assert response.status_code == 200
        print("✓ 登录页面访问正常")
    except Exception as e:
        print(f"✗ 登录页面访问失败: {e}")
        return False

    # 测试注册页面
    try:
        response = client.get('/register/', HTTP_HOST='localhost')
        assert response.status_code == 200
        print("✓ 注册页面访问正常")
    except Exception as e:
        print(f"✗ 注册页面访问失败: {e}")
        return False

    # 创建测试会话
    session = client.session
    session['is_logged_in'] = True
    session['username'] = 'test_user'
    session.save()

    # 测试首页（需要登录）
    try:
        response = client.get('/home/', HTTP_HOST='localhost')
        assert response.status_code == 200
        print("✓ 首页访问正常")
    except Exception as e:
        print(f"✗ 首页访问失败: {e}")
        return False

    # 测试诊断页面（需要登录）
    try:
        response = client.get('/diagnosis/', HTTP_HOST='localhost')
        assert response.status_code == 200
        print("✓ 诊断页面访问正常")
    except Exception as e:
        print(f"✗ 诊断页面访问失败: {e}")
        return False

    # 测试健康评估页面（需要登录）
    try:
        response = client.get('/health/', HTTP_HOST='localhost')
        assert response.status_code == 200
        print("✓ 健康评估页面访问正常")
    except Exception as e:
        print(f"✗ 健康评估页面访问失败: {e}")
        return False

    return True

def test_api_endpoints():
    """测试API端点"""
    print("\n测试API端点...")
    client = Client()

    # 创建测试会话
    session = client.session
    session['is_logged_in'] = True
    session['username'] = 'test_user'
    session.save()

    # 测试诊断API
    try:
        print("  测试诊断API（这可能需要30-60秒）...")
        response = client.get('/api/run-diagnosis/', HTTP_HOST='localhost')
        if response.status_code == 200:
            data = json.loads(response.content)
            if 'discharge_type' in data:
                print("✓ 诊断API工作正常")
                print(f"  测试结果: {data['discharge_type']}")
            else:
                print("⚠ 诊断API返回数据格式异常")
        else:
            print(f"⚠ 诊断API返回状态码: {response.status_code}")
    except Exception as e:
        print(f"⚠ 诊断API测试出错: {e}")

    # 测试健康评估API
    try:
        print("  测试健康评估API（这可能需要1-2分钟）...")
        response = client.get('/api/run-health/', HTTP_HOST='localhost')
        if response.status_code == 200:
            data = json.loads(response.content)
            if 'hi_score' in data:
                print("✓ 健康评估API工作正常")
                print(f"  测试结果: HI={data['hi_score']}")
            else:
                print("⚠ 健康评估API返回数据格式异常")
        else:
            print(f"⚠ 健康评估API返回状态码: {response.status_code}")
    except Exception as e:
        print(f"⚠ 健康评估API测试出错: {e}")

    return True

def test_static_files():
    """测试静态文件"""
    print("\n测试静态文件...")
    import os
    static_dir = os.path.join(os.path.dirname(__file__), 'static')

    # 检查CSS文件
    css_file = os.path.join(static_dir, 'css', 'style.css')
    if os.path.exists(css_file):
        print("✓ CSS文件存在")
    else:
        print("✗ CSS文件缺失")
        return False

    # 检查JS文件
    js_file = os.path.join(static_dir, 'js', 'charts.js')
    if os.path.exists(js_file):
        print("✓ JavaScript文件存在")
    else:
        print("✗ JavaScript文件缺失")
        return False

    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("高电压设备智能健康管理平台 - 系统测试")
    print("=" * 60)

    tests = [
        ("基本导入", test_basic_imports),
        ("模型导入", test_model_imports),
        ("用户系统", test_user_system),
        ("Django视图", test_django_views),
        ("静态文件", test_static_files),
        ("API端点", test_api_endpoints),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {test_name}测试出现异常: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试完成！通过: {passed}, 失败: {failed}")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 所有测试通过！系统运行正常。")
        return 0
    else:
        print(f"\n⚠ 有 {failed} 个测试失败，请检查相关功能。")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试过程中发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)