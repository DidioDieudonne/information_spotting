import cv2
import numpy as np
import os
from matcher import show_matches
from extract_features import extract_and_save_all, extract_sift_descriptors
from indexer import build_index
from query_video import extract_query_descriptors
from matcher import match_query_to_index

doc_folder = "data/documents"
desc_folder = "data/descriptors"
video_path = "data/videos/query.mp4"

# Étape 1 : Extraire et sauvegarder les descripteurs
extract_and_save_all(doc_folder, desc_folder)

# Étape 2 : Indexer les documents
index = build_index(desc_folder)

# Étape 3 : Extraire les descripteurs depuis une vidéo
query_descs = extract_query_descriptors(video_path, frame_step=30)

# Étape 4 : Matcher et voter
votes = match_query_to_index(query_descs, index)

print("Résultats du vote :")
for doc, score in votes:
    print(f"{doc}: {score}")


# Charger image de requête (frame vidéo) et document reconnu
frame = cv2.imread("data/videos/frame_query.png")  # ou récupère une frame dynamiquement
doc_name = votes[0][0] + ".jpg"
doc_img = cv2.imread(os.path.join(doc_folder, doc_name))

# Extraire les descripteurs (tu peux les sauvegarder au préalable aussi)
_, des_frame = extract_sift_descriptors("data/videos/frame_query.png")
des_doc = np.load(os.path.join(desc_folder, doc_name + ".npy"))

# Afficher les correspondances
show_matches(frame, doc_img, des_frame, des_doc)