# Generated by Django 5.1 on 2024-08-18 08:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('identification', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='oeil',
            options={},
        ),
        migrations.AddField(
            model_name='oeil',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to='images'),
        ),
    ]
