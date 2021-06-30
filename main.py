import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/2.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.75)

while True:
    success, img = cap.read()

    # change in img size
    img = cv2.resize(img, (850, 550))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)

            # bounding box from class
            bboxC = detection.location_data.relative_bounding_box
            img_height, img_width, img_channel = img.shape
            bbox = int(bboxC.xmin * img_width), int(bboxC.ymin * img_height), \
                   int(bboxC.width * img_width), int(bboxC.height * img_height)
            cv2.rectangle(img, bbox, (0, 204, 0), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 0.95, (0, 204, 0), 2)

    cv2.imshow("image", img)

    # frame rate calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 10, 100), 2)
    cv2.imshow("IMAGE", img)

    cv2.waitKey(1)  # we can change here to change the fps
