@echo off
:: Lancer le serveur Django
start cmd /k "py manage.py runserver"

:: Attendre que le serveur soit démarré
timeout /t 5 /nobreak >nul

:: Ouvrir le navigateur par défaut sur les URL spécifiées
start http://127.0.0.1:8000/
start http://127.0.0.1:5000/