@echo off
:: Mettre à jour pip
python -m pip install --upgrade pip

:: Installer les dépendances nécessaires
pip install django opencv-python numpy scipy Pillow