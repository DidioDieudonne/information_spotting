import cv2
from extract_features import extract_sift_from_image

def extract_query_descriptors(video_path, frame_step=30):
    cap = cv2.VideoCapture(video_path)
    descriptors_all = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, des = extract_sift_from_image(gray)
            if des is not None:
                descriptors_all.append(des)
        count += 1
    cap.release()
    return descriptors_all

def display_video_frames(video_path, frame_step=30):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_step == 0:
            cv2.imshow("Frame extraite", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        count += 1
    cap.release()
    cv2.destroyAllWindows()