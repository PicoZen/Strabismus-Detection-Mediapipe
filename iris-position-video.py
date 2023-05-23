import cv2
import numpy as np
import mediapipe as mp
import math

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]

L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]

mp_face_mesh = mp.solutions.face_mesh


def euclidian_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance


def iris_position(iris_centre, right_point, left_point):
    center_to_right = euclidian_distance(iris_centre, right_point)
    center_to_left = euclidian_distance(iris_centre, left_point)
    total_distance = euclidian_distance(right_point, left_point)
    ratio1 = center_to_right / center_to_left
    ratio2 = center_to_left / center_to_right
    return ratio1, ratio2


cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # print(results.multi_face_landmarks[0].landmark)
            # print(mesh_points.shape)
            # cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA )
            # cv2.polylines (frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.polylines (frame, [mesh_points[LEFT_IRIS]], True, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.polylines (frame, [mesh_points[RIGHT_IRIS]], True, (0, 255, 0), 1, cv2.LINE_AA)
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle (mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            print(center_left, center_right)

            # eye center loc
            cv2.circle(frame, center_left, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, center_right, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.circle(frame, center_left, int(l_radius), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle (frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle (frame, mesh_points[L_H_LEFT][0], 2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle (frame, mesh_points[L_H_RIGHT][0], 2, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle (frame, mesh_points[R_H_LEFT][0], 2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle (frame, mesh_points[R_H_RIGHT][0], 2, (0, 255, 255), 1, cv2.LINE_AA)

            _, left_ratio = iris_position (center_left, mesh_points[L_H_RIGHT][0], mesh_points[L_H_LEFT][0])
            right_ratio, _ = iris_position (center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0])
            print(left_ratio, right_ratio)

            total_ratio = max(right_ratio, left_ratio) / min(right_ratio, left_ratio)
            print(total_ratio)

            cv2.putText (frame, f"Deviation : {total_ratio:.2f}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("Live", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()