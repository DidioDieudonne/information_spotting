import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path
from extract_features import extract_sift_descriptors
from indexer import build_index
from matcher import match_query_to_index, show_matches, localize_region

# --- Configuration ---
st.set_page_config(page_title="Information Spotting", layout="wide")
doc_folder = "data/documents"
desc_folder = "data/descriptors"
video_folder = "data/videos"

st.title("Information Spotting - Indexation & Requête par Vidéo")

# --- Section 1 : Indexation des documents ---
st.header("1. Indexation des documents")

if st.button("Indexer les documents"):
    from extract_features import extract_and_save_all
    extract_and_save_all(doc_folder, desc_folder)
    st.success("Indexation terminée. Descripteurs sauvegardés.")

# --- Section 2 : Chargement de la requête vidéo ---
st.header("2. Charger et traiter une vidéo")

video_file = st.file_uploader("Choisir une vidéo de requête", type=["mp4", "avi"])
frame_step = st.slider("Extraire une frame toutes les n images", 1, 100, 30)

if video_file is not None:
    tmp_path = Path("temp_video.mp4")
    tmp_path.write_bytes(video_file.read())

    st.video(str(tmp_path))

    from query_video import extract_query_descriptors
    query_descs = extract_query_descriptors(str(tmp_path), frame_step=frame_step)
    st.success(f"{len(query_descs)} frames analysées.")

    cap = cv2.VideoCapture(str(tmp_path))
    ret, frame = cap.read()
    cap.release()
    if ret:
        os.makedirs("data/videos", exist_ok=True)
        cv2.imwrite("data/videos/frame_query.png", frame)
        st.info("Une frame a été enregistrée pour le matching.")
    else:
        st.warning("Impossible de lire une frame pour le matching visuel.")

    # --- Section 3 : Recherche et affichage des résultats ---
    st.header("3. Recherche du document correspondant")
    index = build_index(desc_folder)

    votes = match_query_to_index(query_descs, index)
    if votes:
        st.subheader("Documents les plus similaires :")
        for doc, score in votes[:5]:
            doc_clean = os.path.splitext(doc)[0]
            doc_img_path = os.path.join(doc_folder, doc_clean + ".jpg")
            st.write(f"{doc_clean} : {score} votes")
            if os.path.exists(doc_img_path):
                st.image(doc_img_path, caption=doc_clean, width=300)

        if st.checkbox("Afficher le matching de la meilleure correspondance"):
            best_doc = votes[0][0]
            doc_clean = os.path.splitext(best_doc)[0]
            doc_img_path = os.path.join(doc_folder, doc_clean + ".jpg")
            doc_img = cv2.imread(doc_img_path)

            if doc_img is None:
                st.error(f"L'image document '{doc_img_path}' est introuvable ou vide.")
            else:
                frame = cv2.imread("data/videos/frame_query.png")
                if frame is None:
                    st.error("Frame de requête introuvable.")
                else:
                    _, des_frame = extract_sift_descriptors("data/videos/frame_query.png")
                    desc_path = os.path.join(desc_folder, doc_clean + ".jpg.npy")
                    if not os.path.exists(desc_path):
                        st.error(f"Descripteur introuvable pour {best_doc}")
                    else:
                        des_doc = np.load(desc_path)
                        img_kp = cv2.drawKeypoints(doc_img, cv2.SIFT_create().detect(doc_img, None), None)
                        st.image(img_kp, caption=f"Keypoints de {doc_clean}", channels="BGR", width=500)

                        # Nouvelle partie : REGION SPOTTED
                        kp_doc, _ = cv2.SIFT_create().detectAndCompute(doc_img, None)
                        kp_query, _ = cv2.SIFT_create().detectAndCompute(frame, None)
                        region_img = localize_region(doc_img, kp_doc, des_doc, kp_query, des_frame, frame)
                        if region_img is not None:
                            st.image(region_img, caption="Region Spotted dans le document", channels="BGR", width=500)
                        else:
                            st.warning("La région n’a pas pu être localisée.")

                        # --- Affichage final dans trois colonnes : query, doc, region spotted ---
                        st.header("4. Résultat final : Information Spotted")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.subheader("Query (extrait vidéo)")
                            st.image("data/videos/frame_query.png", caption="Query")

                        with col2:
                            st.subheader("Document Retrieval")
                            st.image(doc_img_path, caption=doc_clean)

                        with col3:
                            st.subheader("Region Spotted")
                            if region_img is not None:
                                st.image(region_img, caption="Region Spotée")
                            else:
                                st.warning("Pas de région localisée.")
    else:
        st.warning("Aucun résultat trouvé. Vérifiez l’index ou les descripteurs de la requête.")