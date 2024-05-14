import mediapipe.python.solutions.hands as mp_hands
import cv2
import os


DATA_DIR = './data'
def testData(dir):
    hands = mp_hands.Hands(
        # static_image_mode=False,
        static_image_mode=True,
        min_tracking_confidence=0.5,
        min_detection_confidence=0.5
    )

    c1 = c2 = 0

    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)


        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
        if len(data_aux) == 84:
            c1 += 1
        elif len(data_aux) == 42:
            c2 += 1

    return 1 < c1 < 50, 1 < c2 < 50, c1 < c2
    # return 1 < c1 < 100, 1 < c2 < 100, c1 < c2


def testImage(results):
    # data_aux = []
    # x_ = []
    # y_ = []
    s = 0
    for hand_landmarks in results.multi_hand_landmarks:
        s += len(hand_landmarks.landmark) * 2
    #
    #     for i in range(len(hand_landmarks.landmark)):
    #         x = hand_landmarks.landmark[i].x
    #         y = hand_landmarks.landmark[i].y
    #         x_.append(x)
    #         y_.append(y)
    #
    #     for i in range(len(hand_landmarks.landmark)):
    #         x = hand_landmarks.landmark[i].x
    #         y = hand_landmarks.landmark[i].y
    #         data_aux.append(x - min(x_))
    #         data_aux.append(y - min(y_))
    #
    # return len(data_aux) == 84

    return s, s == 84