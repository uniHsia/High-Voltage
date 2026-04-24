#!/usr/bin/env python
"""
自动修复脚本
尝试自动修复常见的按钮无反应、图表不显示问题
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"执行命令: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"✅ {description} - 成功")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} - 失败")
            if result.stderr:
                print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False

def main():
    """主修复流程"""
    print("=" * 60)
    print("🚀 高电压设备智能健康管理平台 - 自动修复工具")
    print("=" * 60)

    # 获取项目目录
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)

    print(f"\n📁 项目目录: {project_dir}")

    # 修复步骤
    fixes = [
        (
            "python manage.py collectstatic --noinput",
            "收集静态文件"
        ),
        (
            "python manage.py migrate --run-syncdb",
            "同步数据库"
        ),
        (
            "python check_static_files.py",
            "检查静态文件配置"
        ),
        (
            "python quick_test.py",
            "测试核心功能（可能需要2-3分钟）"
        )
    ]

    success_count = 0
    fail_count = 0

    for command, description in fixes:
        if run_command(command, description):
            success_count += 1
        else:
            fail_count += 1
            print(f"⚠️ {description} 失败，但继续执行...")

    # 总结
    print("\n" + "=" * 60)
    print("📊 修复总结")
    print("=" * 60)
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")

    if fail_count == 0:
        print("\n🎉 所有修复步骤完成！")
        print("\n📋 接下来的步骤:")
        print("1. 清除浏览器缓存 (Ctrl+Shift+Delete)")
        print("2. 启动Django服务器: python manage.py runserver")
        print("3. 访问: http://127.0.0.1:8000/")
        print("4. 登录后访问调试页面: http://127.0.0.1:8000/debug/")
        print("5. 在调试页面测试各项功能")
    else:
        print(f"\n⚠️ 有 {fail_count} 个步骤失败")
        print("建议手动检查或查看详细错误信息")

    # 提供手动修复建议
    print("\n" + "=" * 60)
    print("🔍 手动检查清单")
    print("=" * 60)
    print("如果问题依然存在，请按以下步骤手动检查:")
    print()
    print("1. 📖 打开浏览器开发者工具 (F12)")
    print("2. 🔴 查看Console标签中的红色错误信息")
    print("3. 🌐 查看Network标签中的请求状态")
    print("4. 🖥️  查看Django服务器终端的错误日志")
    print("5. 🧪 访问调试页面: http://127.0.0.1:8000/debug/")
    print("6. 💻 在调试页面运行各项测试")
    print()
    print("详细故障排除指南请查看: STEP_BY_STEP_FIX.md")

    return fail_count == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 修复被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 修复过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)