import os
import time
from test_no_of_features import testData
from test_no_of_features import testImage
import cv2
import shutil
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    static_image_mode=False
)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1
dataset_size = 200
# max_dataset_size = 300

cap = cv2.VideoCapture(0)

j = 25
while number_of_classes > 0:
    # if os.path.exists(os.path.join(DATA_DIR, str(j))):
    #     j += 1
    #     continue
    # if j in [0, 1, 2, 5, 11]:
    #     j += 1
    #     number_of_classes -= 1
    #     continue

    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))
    #     # counter = 0
    # elif len(os.listdir(os.path.join(DATA_DIR, str(j)))) == 0:
    #     pass
    # else:
    #     j += 1
    #     number_of_classes -= 1
    #     continue
    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()

        cv2.flip(frame, 1, frame)
        cv2.putText(frame,
                    "Collecting data for: {}, Press 'Q' when ready, Press 'S' to exit".format(labels_dict[j]),
                    (5, 22),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
        )

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(45, 2, 248), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(29, 255, 64), thickness=2, circle_radius=2)
                )

            cv2.putText(frame,
                        "Features: {}".format(testImage(results)[0]),
                        (500,460),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
            )
        cv2.imshow('frame', frame)

        if cv2.waitKey(20) == ord('q'):
            time.sleep(3)
            break
        elif cv2.waitKey(20) & 0xff == ord('s'):
            exit("Mind Change! End.")

    counter = 100
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)
        ori_frame = frame.copy()
        # cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        cv2.putText(frame,
                    "Collecting data for: {} - Image: {}".format(labels_dict[j], counter),
                    (5, 22),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                    )

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(45, 2, 248), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(29, 255, 64), thickness=2, circle_radius=2)
                )

            cv2.putText(frame,
                        "Features: {}".format(testImage(results)[0]),
                        (500, 460),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
            )

        cv2.imshow('frame', frame)
        cv2.waitKey(100)
        if cv2.waitKey(20) & 0xff == ord('s'):
            exit("Mind Change! End.")

        if results.multi_hand_landmarks:
            if j in [2, 11, 14, 20, 21]:
                if not testImage(results)[1]:
                    cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), ori_frame)
                    counter += 1
            else:
                if testImage(results)[1]:
                    cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), ori_frame)
                    counter += 1

        # cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        # counter += 1

    if j in [2, 11, 14, 20, 21]:
        if not testData(str(j))[1] and testData(str(j))[2]:
            j += 1
            number_of_classes -= 1
        else:
            for file in os.listdir(os.path.join(DATA_DIR, str(j)))[0:100]:
                os.remove(os.path.join(DATA_DIR, str(j), file))
    else:
        if not testData(str(j))[0] and not testData(str(j))[2]:
            j += 1
            number_of_classes -= 1
        else:
            for file in os.listdir(os.path.join(DATA_DIR, str(j)))[0:100]:
                os.remove(os.path.join(DATA_DIR, str(j), file))
    # j += 1
    # number_of_classes -= 1

cap.release()
cv2.destroyAllWindows()
