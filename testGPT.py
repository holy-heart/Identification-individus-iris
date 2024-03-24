import cv2
import os

# Fonction pour extraire les caractéristiques SIFT d'une image
def extract_sift_features(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# Fonction pour faire correspondre les descripteurs SIFT entre une image requête et une base de données
def match_features(query_descriptors, database_descriptors, threshold=50):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(query_descriptors, database_descriptors)
    good_matches = []
    for m in matches:
        if m.distance < threshold:
            good_matches.append([m])
    return good_matches

# Fonction pour charger une image depuis un fichier
def load_image(file_path):
    return cv2.imread(file_path,0)

# Fonction principale pour l'identification de l'iris
def identify_iris(query_image_path, database_images):
    query_image = cv2.imread(query_image_path,0)
    query_descriptors = extract_sift_features(query_image)
    max_matches = 0
    best_match_image = None
    
    for database_image_path in database_images:
        database_image = load_image(database_image_path)
        database_descriptors = extract_sift_features(database_image)
        
        matches = match_features(query_descriptors, database_descriptors)
        num_matches = len(matches)
        print(database_image_path,len(matches))
        
        if num_matches > max_matches:
            max_matches = num_matches
            best_match_image = database_image_path
            
    # Décision basée sur le nombre de bonnes correspondances    
    if max_matches > SEUIL:
        print("Personne identifiée ",query_image_path,": pour un score matche de", max_matches, " avec la photo", best_match_image)
    else:
        print("Personne non reconnue")

# Paramètres
SEUIL = 20  # Ajustez le seuil selon vos besoins

# Base de données d'images
database_images = []
for file in sorted(os.listdir("database1")):
    database_images.append(os.path.join("database1", file))

# Chemin vers l'image requête
query_image_path = "009L_3.png"  

# Identifier l'iris
identify_iris(query_image_path, database_images)