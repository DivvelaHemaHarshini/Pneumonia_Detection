from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class SoundPredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    audio_file = models.FileField(upload_to='lung_sounds/')
    prediction_result = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username} - {self.prediction_result}'