from django.db import models
class Pic(models.Model):
    pic=models.ImageField(upload_to='static/images/', default='default.png', max_length=200, blank=True, null=True, verbose_name='待分类图像')
    # category = models.IntegerField(default=0)
    # accuracy = models.FloatField(default=0.)
    def __str__(self):
        return self.name
    # class Meta作为嵌套类，给上级类添加一些功能
    class Meta:
        #按创建时间的反序排列，最近的最先显示
        ordering = ['-id']
        # 模型类命名
        verbose_name = "图像"
        # 模型类复数命名，若不指定Django会自动在模型名称后加一个’s’
        verbose_name_plural = "图像"
    objects = models.Manager()
