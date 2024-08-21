from django.db import models

class Oeil(models.Model):

    image = models.ImageField(null=True, blank=True, upload_to="images")

    