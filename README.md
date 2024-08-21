...
# Identification par l'iris de l'oeil avec Python et Opencv


...
## Pojet consistant faire matcher deux yeux identiques a partir d'une base de données, en utilisant la technologie Opencv et Django pour l'insterface!

Projet d’identification par l’iris de l'œil. La réalisation de ce projet a été faite avec la bibliothèque Opencv qui est spécialisée dans le traitement d’image, utile donc pour la comparaison des matches entre une photo insérée et la base de données. Pour l’interface, j’ai choisi le framework Django. Nous avons une base de données qui contient 3 photos de chaque œil de 64 personnes, donc 384 images.

1. Installer les packages django, opencv-python, numpy, scipy et Pillow
2. Effectuer les migration de la Base de donnée py manage.py makemigrations && py manage.py migrate
3. activer le serveur avec py manage.py runserver, le lien sera affiché sur le terminal
4. Cas de reussite : Extraire une image de la base de données, puis l’introduire dans la page d'accueille
5. Cas d'echec : Insérer n’importe quelle photo