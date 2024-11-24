from django.db import models

# Create your models here.
class RecognitionHistory(models.Model):
    recognized_id = models.CharField(max_length=100)
    recognized_name = models.CharField(max_length=255)
    recognition_time = models.DateTimeField(auto_now_add=True)
    prediction_accuracy = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.recognized_id} - {self.recognized_name} at {self.recognition_time} with {self.prediction_accuracy}% accuracy"


class IdNameMapping(models.Model):
    id_code = models.CharField(max_length=100)
    name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.id_code} - {self.name}"
