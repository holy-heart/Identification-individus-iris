from django.shortcuts import render, redirect
from django.conf import settings
from .forms import OeilForm
import cv2
import os
import numpy as np
from scipy.spatial.distance import euclidean

# Fonction qui retourne les keypoints de l'iris de chaque image
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    K = []
    D = []
    for i, j in zip(keypoints, descriptors):
        x, y = i.pt
        distance = euclidean((x, y), (383, 287))
        if distance < 250 and distance > 70:
            K.append(i)
            D.append(j)
    return K, D

# Fonction qui retourne les bons matches entre deux images
def match_features(probleme_d, database_d, ration_distance=0.76):
    bf = cv2.BFMatcher()
    probleme = np.array(probleme_d)
    database = np.array(database_d)
    matches = bf.knnMatch(probleme, database, k=2)
    good_matches = []
    # si les deux points se ressemble, alors le premier matche est un outlier
    for m, n in matches:
        if (m.distance / n.distance) < ration_distance:
            good_matches.append(m)
    return good_matches

def home(request):
    if request.method == "POST":
        form = OeilForm(request.POST, request.FILES)
        if form.is_valid():
            file_path = os.path.join(settings.MEDIA_ROOT, 'images/probleme.png')

            # Vérifier si le fichier existe avant de le supprimer
            if os.path.exists(file_path):
                os.remove(file_path)
            image = form.save(commit=False)
            image.image.name = 'probleme.png'
            image.save()
            return redirect('solution')
    else:
        form = OeilForm()
    return render(request, 'identification/ident.html', {'form': form})

def solution(request):
    SEUIL = 100
    file_name = 'images/probleme.png'
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)
    
    # Vérifier si le fichier existe avant de continuer
    if not os.path.exists(file_path):
        return render(request, "identification/refu.html")

    probleme_c = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)# En gris
    probleme_c = cv2.equalizeHist(probleme_c) # Contraste
    cv2.imwrite(file_path, probleme_c)
    
    probleme_k, probleme_d = extract_sift_features(probleme_c)
    
    # Liste des fichiers dans le répertoire media
    database_images = []
    for file in sorted(os.listdir(os.path.join(settings.BASE_DIR, 'database1'))):
        database_images.append(os.path.join("database1", file))

    max_matches, match_sol, solution, solution_c, solution_k = 0, [], None, None, None
    # On compare notre image avec chauque image du database
    for database in database_images:
        # Traitement de chauqe image du database
        image_c2 = cv2.imread(database, cv2.IMREAD_GRAYSCALE)
        database_c = cv2.equalizeHist(image_c2)
        database_k, database_d = extract_sift_features(database_c)
        # Matche
        matches = match_features(probleme_d, database_d)
        print(database,len(matches))
        # On prend celle qui a le plus de matches
        if len(matches) > max_matches:
            match_sol, max_matches = matches, len(matches)
            solution, solution_c, solution_k = database, database_c, database_k

    if len(match_sol) > SEUIL: #En general il y a plus de 100 matches pour les vrais matches
        comp = cv2.drawMatches(probleme_c, probleme_k, solution_c, solution_k, match_sol, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        result_path = os.path.join(settings.MEDIA_ROOT, 'result.png')
        cv2.imwrite(result_path, comp)
        id_ = os.path.basename(solution)[:3]
        eye = os.path.basename(solution)[3:4]
        return render(request, "identification/sol.html", {"id": id_, "eye": eye, "pic": os.path.basename(solution)})
    else:
        return render(request, "identification/refu.html")   