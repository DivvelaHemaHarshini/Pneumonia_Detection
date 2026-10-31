from django.db import models

# Create your models here.
# classifier/models.py

from django.db import models
from django.contrib.auth.models import User

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='history_pics/')
    prediction_result = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username} - {self.prediction_result} at {self.timestamp.strftime("%Y-%m-%d %H:%M")}'
    