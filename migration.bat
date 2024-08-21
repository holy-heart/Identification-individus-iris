@echo off
:: Appliquer les migrations pour configurer la base de données
py manage.py makemigrations
py manage.py migrate