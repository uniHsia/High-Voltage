#!/usr/bin/env python
"""
静态文件检查脚本
验证静态文件配置是否正确
"""
import os
import sys
from pathlib import Path

def check_static_files():
    """检查静态文件配置"""
    print("=" * 60)
    print("静态文件配置检查")
    print("=" * 60)

    # 获取项目根目录
    project_root = Path(__file__).resolve().parent
    static_dir = project_root / 'static'

    print(f"\n项目根目录: {project_root}")
    print(f"静态文件目录: {static_dir}")

    # 检查静态文件目录是否存在
    if not static_dir.exists():
        print(f"✗ 静态文件目录不存在: {static_dir}")
        return False
    print(f"✓ 静态文件目录存在")

    # 检查关键文件
    required_files = [
        static_dir / 'css' / 'style.css',
        static_dir / 'js' / 'charts.js',
    ]

    all_exist = True
    for file_path in required_files:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ {file_path.relative_to(project_root)} 存在 (大小: {size} 字节)")
        else:
            print(f"✗ {file_path.relative_to(project_root)} 不存在")
            all_exist = False

    if not all_exist:
        return False

    # 检查Django配置
    print("\n检查Django配置...")
    try:
        import django
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'high_voltage_platform.settings')
        django.setup()

        from django.conf import settings
        from django.contrib.staticfiles.finders import find

        print(f"✓ Django设置模块: {settings.SETTINGS_MODULE}")
        print(f"✓ STATIC_URL: {settings.STATIC_URL}")
        print(f"✓ STATICFILES_DIRS: {settings.STATICFILES_DIRS}")

        # 测试静态文件查找
        print("\n测试静态文件查找...")
        try:
            charts_js_path = find('js/charts.js')
            print(f"✓ charts.js 可以通过staticfiles找到: {charts_js_path}")
        except Exception as e:
            print(f"✗ charts.js 无法通过staticfiles找到: {e}")
            return False

        try:
            style_css_path = find('css/style.css')
            print(f"✓ style.css 可以通过staticfiles找到: {style_css_path}")
        except Exception as e:
            print(f"✗ style.css 无法通过staticfiles找到: {e}")
            return False

    except Exception as e:
        print(f"✗ Django配置检查失败: {e}")
        return False

    # 检查charts.js内容
    print("\n检查charts.js文件内容...")
    charts_js = static_dir / 'js' / 'charts.js'
    try:
        content = charts_js.read_text(encoding='utf-8')

        # 检查关键函数
        required_functions = [
            'renderPRPDInputChart',
            'renderProbabilityChart',
            'renderGaugeChart',
            'renderFusionTrendChart',
            'renderAnomalyTrendChart',
            'renderRULChart',
            'renderHomeTrendChart',
            'renderHistoryTrendChart'
        ]

        missing_functions = []
        for func in required_functions:
            if f'function {func}' in content or f'{func}=' in content:
                print(f"✓ 函数 {func} 存在")
            else:
                print(f"✗ 函数 {func} 缺失")
                missing_functions.append(func)

        if missing_functions:
            print(f"\n✗ 缺少函数: {missing_functions}")
            return False

        # 检查事件监听器
        if 'addEventListener' in content:
            print("✓ 包含事件监听器代码")
        else:
            print("✗ 缺少事件监听器代码")
            return False

    except Exception as e:
        print(f"✗ 读取charts.js失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ 静态文件配置检查通过！")
    print("=" * 60)
    print("\n建议操作：")
    print("1. 清除浏览器缓存（Ctrl+Shift+Delete）")
    print("2. 重启Django服务器")
    print("3. 检查浏览器控制台是否有JavaScript错误")

    return True

if __name__ == "__main__":
    try:
        success = check_static_files()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n检查过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)