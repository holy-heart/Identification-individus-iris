import cv2
import os
import numpy as np

# Fonction pour extraire les caractéristiques SIFT d'une image
def extract_sift_features(image, diametre_cercle=100):
    # Créer un détecteur SIFT
    sift = cv2.SIFT_create()
    
    # Détecter les keypoints et calculer les descripteurs
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    # Trouver les dimensions de l'image
    hauteur, largeur = image.shape[:2]
    
    # Trouver le centre de l'image
    centre_x = largeur // 2
    centre_y = hauteur // 2
    
    # Liste pour stocker les nouveaux keypoints à l'intérieur du cercle
    nouveaux_keypoints = []
    
    # Parcourir tous les keypoints détectés
    for kp in keypoints:
        # Calculer la distance entre le keypoint et le centre de l'image
        distance = ((kp.pt[0] - centre_x) ** 2 + (kp.pt[1] - centre_y) ** 2) ** 0.5
        # Vérifier si le keypoint est à l'intérieur du cercle
        if distance <= diametre_cercle / 2:
            nouveaux_keypoints.append(kp)
    
    # Convertir la liste de nouveaux keypoints en un tableau numpy
    nouveaux_keypoints = np.array([kp.pt for kp in nouveaux_keypoints])
    
    # Extraire les descripteurs correspondant uniquement aux keypoints à l'intérieur du cercle
    indices_descripteurs = np.where([kp.pt in nouveaux_keypoints for kp in keypoints])[0]
    nouveaux_descripteurs = descriptors[indices_descripteurs, :]
    # Retourner les nouveaux keypoints et descripteurs
    return nouveaux_keypoints, nouveaux_descripteurs

def match_features(probleme_d, database_d, threshold=150):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(probleme_d, database_d)
    good_matches = []
    for m in matches:
        if m.distance < threshold:
            good_matches.append(m)
    return good_matches



# Paramètres
SEUIL = 5  # Ajustez le seuil selon vos besoins

# Base de données d'images
database_images = []
for file in sorted(os.listdir("database1")):
    database_images.append(os.path.join("database1", file))

# Chemin vers l'image requête
probleme = "051L_1.png"  


probleme_c = cv2.imread(probleme,1)
probleme_k, probleme_d = extract_sift_features(probleme_c)
max_matches = 0

for database in database_images:
    database_c = cv2.imread(database,1)
    database_k, database_d = extract_sift_features(database_c)
    
    matches = match_features(probleme_d, database_d)
    print(database,len(matches))
    
    if len(matches) > max_matches:
        match_sol= matches
        max_matches = len(matches)
        solution = database
        solution_c = database_c
        solution_k = database_k
            
    # Décision basée sur le nombre de bonnes correspondances    
if max_matches > SEUIL:
    print("Personne identifiée ",probleme,": pour un score matche de", max_matches, " avec la photo", solution)
else:
    print("Personne non reconnue")

comp = cv2.drawMatches(probleme_c,probleme_k,solution_c, solution_k, match_sol, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('comparaison', comp)
cv2.waitKey(0)
cv2.imwrite("result.png", comp)