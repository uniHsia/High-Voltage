#!/usr/bin/env python
"""
数据库初始化脚本
用于初始化数据库和创建默认管理员用户
"""
import os
import sys
import django

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'high_voltage_platform.settings')
django.setup()

from django.contrib.auth import get_user_model
from dashboard.utils import register_user, user_exists

def init_database():
    """初始化数据库"""
    print("=" * 60)
    print("高电压设备智能健康管理平台 - 数据库初始化")
    print("=" * 60)

    # 创建默认管理员用户
    admin_username = "admin"
    admin_email = "admin@highvoltage.com"
    admin_password = "admin123"

    print(f"\n检查管理员用户...")
    if not user_exists(admin_username):
        print(f"创建默认管理员用户...")
        try:
            register_user(admin_username, admin_email, admin_password)
            print(f"✓ 管理员用户创建成功！")
            print(f"  用户名: {admin_username}")
            print(f"  密码: {admin_password}")
            print(f"  邮箱: {admin_email}")
        except Exception as e:
            print(f"✗ 创建管理员用户失败: {e}")
            return False
    else:
        print(f"✓ 管理员用户已存在，跳过创建")

    print("\n" + "=" * 60)
    print("数据库初始化完成！")
    print("=" * 60)
    print("\n您现在可以使用以下凭据登录:")
    print(f"  用户名: {admin_username}")
    print(f"  密码: {admin_password}")
    print(f"\n请登录后立即修改默认密码！")
    print("=" * 60)

    return True

if __name__ == "__main__":
    try:
        success = init_database()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n初始化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)