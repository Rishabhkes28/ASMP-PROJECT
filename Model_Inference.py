import os.path
import pickle

import mediapipe as mp
import cv2
import numpy as np

if os.path.exists('./model1.p'):
    model_dict_1 = pickle.load(open('./model1.p', 'rb'))
    model1 = model_dict_1['model1']
if os.path.exists('./model2.p'):
    model_dict_2 = pickle.load(open('./model2.p', 'rb'))
    model2 = model_dict_2['model2']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3
)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}
# labels_dict = {0: 'B', 1: 'O', 2: 'A'}
# labels_dict = {0: '1', 1: '2', 2: '3', 3: '4', 4: '10'}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    # cv2.flip(frame, 1, frame)
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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
            # print(len(data_aux))
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        if len(data_aux) == 42 and os.path.exists('model1.p'):
            prediction = model1.predict([np.asarray(data_aux)])
        elif len(data_aux) == 84 and os.path.exists('model2.p'):
            prediction = model2.predict([np.asarray(data_aux)])
        else:
            continue
        # if int(prediction) not in labels_dict:
        #     labels_dict[prediction] = 'Please enter key'

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
