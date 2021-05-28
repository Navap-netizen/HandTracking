import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # we can mentions parameters or it will take the parameters which are default
mpDraw = mp.solutions.drawing_utils  # To draw the hands

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Process methods only works with RGB images
    results = hands.process(imgRGB)
    # Extracting the information(MultiHands) from results
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # handLms == handLandmarks
            #  getting info' from these hands like id, landmark, ..
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape  # height, width, channel
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                # drawing circles at the channels position
                # if id == 0:
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # drawing the info' of (single) hands
            # mpHands.HAND_CONNECTIONS will draw the connections between landmarks

    cTime = time.time()  # give us the current time
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),
                2)  # img, text, place, fontFace, fontScale, color, thickness

    cv2.imshow("Image", img)
    cv2.waitKey(1)
