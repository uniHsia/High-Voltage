from django.db import models


class Device(models.Model):
    name = models.CharField(max_length=100, verbose_name='设备名称')
    health_score = models.FloatField(default=0.0, verbose_name='健康分数')
    status = models.CharField(max_length=50, default='正常', verbose_name='设备状态')
    is_online = models.BooleanField(default=True, verbose_name='是否在线')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    def __str__(self):
        return self.name


class AlarmRecord(models.Model):
    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='alarms')
    level = models.CharField(max_length=50, verbose_name='告警等级')
    message = models.TextField(verbose_name='告警内容')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    def __str__(self):
        return f'{self.device.name} - {self.level}'