import cv2
import os

# Fonction pour extraire les caractéristiques SIFT d'une image
def extract_sift_features(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(probleme_d, database_d, threshold=50):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(probleme_d, database_d)
    good_matches = []
    for m in matches:
        if m.distance < threshold:
            good_matches.append(m)
    return good_matches



# Paramètres
SEUIL = 20  # Ajustez le seuil selon vos besoins

# Base de données d'images
database_images = []
for file in sorted(os.listdir("database1")):
    database_images.append(os.path.join("database1", file))

# Chemin vers l'image requête
probleme = "013L_1.png"  


probleme_c = cv2.imread(probleme,0)
probleme_k, probleme_d = extract_sift_features(probleme_c)
max_matches = 0

for database in database_images:
    database_c = cv2.imread(database,0)
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