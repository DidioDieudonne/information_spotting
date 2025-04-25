
import cv2
import numpy as np
from collections import Counter

def match_query_to_index(query_descs, index):
    votes = {}
    for q_desc in query_descs:
        bf = cv2.BFMatcher()
        for doc_name, doc_desc in index.items():
            matches = bf.knnMatch(q_desc, doc_desc, k=2)
            good = [m for match in matches if len(match) == 2 and match[0].distance < 0.75 * match[1].distance for m in [match[0]]]
            votes[doc_name] = votes.get(doc_name, 0) + len(good)
    return sorted(votes.items(), key=lambda x: x[1], reverse=True)

def show_matches(query_img, doc_img, des_q, des_d):
    sift = cv2.SIFT_create()
    kp_q, _ = sift.detectAndCompute(query_img, None)
    kp_d, _ = sift.detectAndCompute(doc_img, None)

    flann = cv2.FlannBasedMatcher_create()
    matches = flann.knnMatch(des_q, des_d, k=2)
    good = [m for match in matches if len(match) == 2 and match[0].distance < 0.7 * match[1].distance for m in [match[0]]]

    match_img = cv2.drawMatches(query_img, kp_q, doc_img, kp_d, good, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matching", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def localize_region(doc_img, kp_doc, des_doc, kp_query, des_query, frame, ratio_thresh=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_doc, k=2)

    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    if len(good_matches) >= 4:
        src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_doc[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            h, w = frame.shape[:2]
            corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
            projected = cv2.perspectiveTransform(corners, H)
            doc_img_out = doc_img.copy()
            cv2.polylines(doc_img_out, [np.int32(projected)], True, (255,0,0), 4)
            return doc_img_out
    return None