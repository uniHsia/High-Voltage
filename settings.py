# ── 项目结构说明 ────────────────────────────────────────────
#
# myproject/                  ← Django 项目根目录
# ├── manage.py
# ├── myproject/
# │   ├── settings.py
# │   └── urls.py             ← 在这里 include hvms.urls
# └── hvms/                   ← 本 App 目录（复制到此处）
#     ├── views.py
#     ├── urls.py
#     ├── templates/
#     │   └── hvms/
#     │       └── index.html
#     └── static/
#         ├── css/
#         │   └── style.css
#         └── js/
#             └── charts.js
#
# ── settings.py 需要的关键配置 ──────────────────────────────

SETTINGS_SNIPPET = """
INSTALLED_APPS = [
    ...
    'hvms',          # 添加 app
]

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,   # 启用 APP_DIRS 自动查找 templates/
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

STATIC_URL = '/static/'
# 开发时 Django 会自动从每个 App 的 static/ 目录提供静态文件
"""

# ── myproject/urls.py ──────────────────────────────────────

PROJECT_URLS = """
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('hvms.urls')),      # 根路径映射到 hvms
]
"""

# ── 启动命令 ──────────────────────────────────────────────
STARTUP = """
python manage.py migrate
python manage.py runserver
# 访问 http://127.0.0.1:8000/
"""