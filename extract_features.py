import cv2
import os
import numpy as np

def extract_sift_descriptors(image_path):
    sift = cv2.SIFT_create()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"[ERREUR] L'image '{image_path}' est vide ou introuvable.")

    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def extract_sift_from_image(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def extract_and_save_all(doc_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for file in os.listdir(doc_folder):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            kp, des = extract_sift_descriptors(os.path.join(doc_folder, file))
            if des is not None:
                np.save(os.path.join(save_folder, file + ".npy"), des)

def show_keypoints(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, _ = sift.detectAndCompute(gray, None)
    img_kp = cv2.drawKeypoints(image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints - " + os.path.basename(image_path), img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()