from django.urls import path
from . import views

app_name = 'hvms'

urlpatterns = [
    path('', views.index, name='index'),
]