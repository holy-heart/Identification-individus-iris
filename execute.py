from flask import Flask, redirect, url_for, render_template, request, flash
from scipy.spatial.distance import euclidean
import cv2
import os
import numpy as np
#pip install Flask opencv-python-headless scipy numpy
#il suffit d'executer ce script python et d'entrer l'url qui s'affiche dans le terminal sur un navigateur

#Fonction qui retourne les keypoints de l'iris de chaque image
def extract_sift_features(image):
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


#fonction qui retourne les bons matches entre deux images
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
app.secret_key = 'hmmm'


#page d'accueil, la focntion recoit l'image
@app.route("/", methods=['POST', "GET"])
def home():
    if request.method == "POST":
        image= request.files['img']
        image= image.read()

        with open('static/probleme.png', "wb") as f:
            f.write(image)
        return redirect(url_for("solution"))
    else:
        return render_template("ident.html")


#page Enregistrer une image dans la bdd
@app.route("/insert", methods=['POST', "GET"])
def insert():
    if request.method == "POST":
        image_i=request.files['img_i']
        image_i=image_i.read()

        with open(f'database1/sign.png', "wb") as f:
            f.write(image_i)
        flash("La photo a été ajoutée à la base de données avec succès. Vous pouvez l'identifier sur la page d'accueil, en haut à gauche, avec une autre photo du même œil (ident.png).", "success")
        return render_template("insert.html")
        
    else:
        return render_template("insert.html")



#code pour l'identification
@app.route("/solution/")
def solution():
    
    SEUIL = 100 
    probleme_c = cv2.imread("static/probleme.png",cv2.IMREAD_GRAYSCALE)
    probleme_c = cv2.equalizeHist(probleme_c)#augmente le contrast
    cv2.imwrite("static/probleme.png", probleme_c)#la bibliotheque flask demande l'utilisation d'un fichier static
    #bdd
    database_images = []
    for file in sorted(os.listdir("database1")):
        database_images.append(os.path.join("database1", file))

    probleme_k, probleme_d = extract_sift_features(probleme_c)
    
    max_matches = 0
    #comparer l'image inséré avec toutes les images de database1
    for database in database_images:
        image_c2 = cv2.imread(database,cv2.IMREAD_GRAYSCALE)
        database_c = cv2.equalizeHist(image_c2)
        database_k, database_d = extract_sift_features(database_c)
        matches = match_features(probleme_d, database_d)
        print(database,len(matches))
        #on garde la photo avec le maximum de matches
        if len(matches) > max_matches:
            match_sol= matches
            max_matches=len(match_sol)
            solution = database
            solution_c = database_c
            solution_k = database_k

    if len(match_sol) > SEUIL:
        print("Personne identifiée avec un score de correspondance de", len(match_sol), " avec la photo", solution)
        comp = cv2.drawMatches(probleme_c,probleme_k,solution_c, solution_k, match_sol, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("static/result.png", comp)
        if len(os.path.basename(solution))==10:
            return render_template("sol.html", id=str(os.path.basename(solution))[:3], eye=str(os.path.basename(solution))[3:4], pic=str(os.path.basename(solution)))
        else : 
            return render_template("sol.html", id=65, eye='R', pic=str(os.path.basename(solution)))
    else:
        print("Personne non reconnue")
        return render_template('refu.html')



if __name__ == '__main__':
    app.run(debug=True)