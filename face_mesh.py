import cv2
import mediapipe as mp

LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

image = cv2.imread("face.jpg")

height, width, _ = image.shape

rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = face_mesh.process(rgb_img)
mesh_points = []
for facial_landmarks in results.multi_face_landmarks:
    for i in range(0, 478):
        pt = facial_landmarks.landmark[i]
        x = int(pt.x * width)
        y = int(pt.y * height)
        if i == 468 or i == 473:
            continue
        cv2.circle(image, (x,y), 2, (100, 200, 10), -1)
        mesh_points.append([x,y])

# for i in LEFT_EYE+LEFT_IRIS:
#     cv2.circle(image, mesh_points[i], 2, (100, 200, 10), -1)
#     # cv2.putText(image, str(i), (mesh_points[i]), 0, 0.2, (0, 0, 255))
#
# for j in RIGHT_EYE+RIGHT_IRIS:
#     cv2.circle(image, mesh_points[j], 2, (100, 200, 10), -1)
#     # cv2.putText (image, str (j), (mesh_points[j]), 0, 0.2, (0, 0, 255))


cv2.imshow("Image",image)
cv2.imwrite("face_mesh.jpg",image)
cv2.waitKey(0)