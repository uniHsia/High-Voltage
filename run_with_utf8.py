"""
Windows兼容启动脚本
解决中文显示乱码问题
"""
import sys
import io
import os

# 设置stdout编码为UTF-8（解决Windows GBK问题）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 设置Python环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 启动Django
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from django.core.management import execute_from_command_line
    execute_from_command_line(['manage.py', 'runserver'])
