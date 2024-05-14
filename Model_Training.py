import os
import pickle
import collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import shutil


data_dict = pickle.load(open('./data.pickle', 'rb'))


d1 = []
d2 = []
l1 = []
l2 = []
flag1 = False
flag2 = False

for i in range(len(data_dict['data'])):
    if len(data_dict['data'][i]) == 42:
        d1.append(data_dict['data'][i])
        l1.append(data_dict['labels'][i])
        flag1 = True
    elif len(data_dict['data'][i]) == 84:
        d2.append(data_dict['data'][i])
        l2.append(data_dict['labels'][i])
        flag2 = True
    else:
        exit(-1)


if not flag1 and os.path.exists('model1.p'):
    os.remove('model1.p')
if not flag2 and os.path.exists('model2.p'):
    os.remove('model2.p')


if flag1:
    data1 = np.asarray(d1)
    labels1 = np.asarray(l1)

    print(collections.Counter(labels1))

    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data1, labels1, test_size=0.2, shuffle=True, stratify=labels1)

    model1 = RandomForestClassifier()

    model1.fit(x_train_1, y_train_1)

    y_predict_1 = model1.predict(x_test_1)

    score_1 = accuracy_score(y_predict_1, y_test_1)
    print('{}% of samples were classified correctly !'.format(score_1 * 100))

    f1 = open('model1.p', 'wb')
    pickle.dump({'model1': model1}, f1)
    f1.close()


if flag2:
    data2 = np.asarray(d2)
    labels2 = np.asarray(l2)

    print(collections.Counter(labels2))

    # leastPopulated = [data_2 for d in set(list(labels_2)) for data_2 in list(labels_2) if data_2 == d].count(
    #     min([data_2 for d in set(list(labels_2)) for data_2 in list(labels_2) if data_2 == d],
    #         key=[data_2 for d in set(list(labels_2)) for data_2 in list(labels_2) if data_2 == d].count))
    # print(leastPopulated)

    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(data2, labels2, test_size=0.2, shuffle=True, stratify=labels2)

    model2 = RandomForestClassifier()

    model2.fit(x_train_2, y_train_2)

    y_predict_2 = model2.predict(x_test_2)

    score_2 = accuracy_score(y_predict_2, y_test_2)
    print('{}% of samples were classified correctly !'.format(score_2 * 100))

    f2 = open('model2.p', 'wb')
    pickle.dump({'model2': model2}, f2)
    f2.close()
