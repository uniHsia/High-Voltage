from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('home/', views.home, name='home'),
    path('diagnosis/', views.diagnosis, name='diagnosis'),
    path('health/', views.health, name='health'),
    path('history/', views.history, name='history'),
    path('history/report/preview/', views.report_preview, name='report_preview'),
    path('history/report/export/', views.report_export, name='report_export'),
    path('api/run-diagnosis/', views.run_diagnosis_model, name='run_diagnosis'),
    path('api/run-health/', views.run_health_model, name='run_health'),
    path('api/test-import/', views.test_import, name='test_import'),
    path('api/select-device/', views.select_device, name='select_device'),
    path('api/get-device-data/', views.get_device_data, name='get_device_data'),
    path('test-page/', views.test_page, name='test_page'),
]