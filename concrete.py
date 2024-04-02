from flask import Flask, redirect, url_for, render_template, request
from scipy.spatial.distance import euclidean
import cv2
import os
import numpy as np



# Fonction pour extraire les caractéristiques SIFT d'une image
def extract_sift_features(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    K=[]
    D=[]
    for i, j in zip(keypoints, descriptors):
        x, y = i.pt
        distance = euclidean((x, y), (383, 287))
        if distance < 250 and distance > 70 :
            K.append(i)
            D.append(j)
    return K, D

def match_features(probleme_d, database_d, ration_distance=0.76):
    bf = cv2.BFMatcher()
    probleme = np.array(probleme_d)
    database = np.array(database_d)
    matches = bf.knnMatch(probleme, database, k=2)
    good_matches = []
    for m, n in matches:
        if (m.distance/n.distance) < ration_distance:
            good_matches.append(m)
    return good_matches




app=Flask(__name__, static_url_path='/static')


@app.route("/", methods=['POST', "GET"])
def home():
    if request.method == "POST":
        image= request.files['img']
        image= image.read()
        with open('static/probleme.png', "wb") as f:
            f.write(image)
        return redirect(url_for("solution"))
    else:
        return render_template("insert.html")


@app.route("/solution")
def solution():
# Paramètres
    
    SEUIL = 5  # Ajustez le seuil selon vos besoins
    image_c = cv2.imread("static/probleme.png",cv2.IMREAD_GRAYSCALE)
    probleme_c = cv2.equalizeHist(image_c)
    cv2.imwrite("static/probleme.png", probleme_c)
    # Base de données d'images
    database_images = []
    for file in sorted(os.listdir("database1")):
        database_images.append(os.path.join("database1", file))

    probleme_k, probleme_d = extract_sift_features(probleme_c)
    max_matches = 0

    for database in database_images:
        image_c2 = cv2.imread(database,cv2.IMREAD_GRAYSCALE)
        database_c = cv2.equalizeHist(image_c2)
        database_k, database_d = extract_sift_features(database_c)

        matches = match_features(probleme_d, database_d)
        print(database,len(matches))

        if len(matches) > max_matches:
            match_sol= matches
            max_matches=len(match_sol)
            solution = database
            solution_c = database_c
            solution_k = database_k

        # Décision basée sur le nombre de bonnes correspondances    
    if len(match_sol) > SEUIL:
        print("Personne identifiée pour un score matche de", len(match_sol), " avec la photo", solution)
        comp = cv2.drawMatches(probleme_c,probleme_k,solution_c, solution_k, match_sol, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("static/result.png", comp)
        return render_template("sol.html", x=str(os.path.basename(solution))[:4])
    else:
        print("Personne non reconnue")
        return render_template('refu.html')



if __name__ == '__main__':
    app.run(debug=True)
