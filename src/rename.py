# from __future__ import division, print_function, unicode_literals
from numpy import load
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import glob
import os
#
# data = load('./exc/img_embeddings.npz')
# trainX, labelTrain, testX, labelTest = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# labelTrain = []
# for i in range(1,len(trainy) + 1):
#     labelTrain.append(i)
# print(labelTrain)
# labelTest = []
# for i in range(0,len(testy)):
#     e = i // 20 + 1
#     print(e)
#     labelTest.append(e)
# print(labelTest)
# print(range(0,len(testy)))

# x = trainX[0]
# y = testX[19]
# a = np.matmul(np.transpose(x), y)
#         # nhân phần tử => sau đó tình tổng các phần tử
# b = np.sum(np.multiply(x, x))
# c = np.sum(np.multiply(y, y))
# print(1 - (a / (np.sqrt(b) * np.sqrt(c))))

# def test(x,y):
#     i = 0
#     j = 0
#     while j < 20:
#         a = np.matmul(np.transpose(x), y[j])
#         # nhân phần tử => sau đó tình tổng các phần tử
#         b = np.sum(np.multiply(x, x))
#         c = np.sum(np.multiply(y[j], y[j]))
#         d = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
#         print(d)
#         if d < 0.4:
#             i += 1
#         j = j + 1
#     return i
# print(test(trainX[1], testX))
#
# #
# # print(trainy)
# # print(testX[0])
#
# # cosine
#
#
# # test
# print(labelTrain)
# logreg = linear_model.LogisticRegression(C=1e5,
#         solver = 'lbfgs', multi_class = 'multinomial')
# logreg.fit(trainX, labelTrain)
# y = logreg.predict(testX)
# print("Accuracy: {} %".format(round(accuracy_score(labelTest,y.tolist()) * 100, 2)))
#
all_file_image = glob.glob('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src//Image/Counting/*.jpg')

# all_file_test = glob.glob('Trieu_Vy_TEST/*.jpg')
x = 0
# print(all_file_test)




f = open("C:/Users/SV_Guest/Desktop/Kieu/DoAn/src//Image/stext.txt", 'w')


for i in all_file_image:
    # t = "Count000%d.txt" %(x)
    x = i.split("\\")[-1]
    f.write(x + '\n')
    print(x)