from keras.models import load_model
from numpy import load
import numpy as np
import csv

# data = load('./exc/dataset_compare_embeddings.npz')
# trainX, label, nameFace = data['arr_0'], data['arr_1'], data['arr_2']
# print('Loaded: ', trainX.shape, label.shape)
# print(label)

data = load('./test_dataset_embeddings.npz')
trainX, label, nameFace = data['arr_0'], data['arr_1'], data['arr_2']


def findCosineDistance(img1, img2):
    a = np.matmul(np.transpose(img1), img2)
    b = np.sum(np.multiply(img1, img1))
    c = np.sum(np.multiply(img2, img2))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def comapre(img1, img2):
    cosine_similarity = findCosineDistance(img1, img2)
    # print(cosine_similarity)
    return cosine_similarity



img1 = trainX[0]
k = 0
x = []
for i in range(label.shape[0] - 1):
	if label[i + 1] == label[i]: #and k != label[i]:
		eror_rate = comapre(trainX[i], trainX[i + 1])
		print(label[[i]])
		print('Compare {} and {} error_rate: {}'.format(nameFace[i], nameFace[i + 1], eror_rate))
		# k = label[i]
	# if k == 50:
	# 	break

#er dataset
# img1 = trainX[0]
# k = 0
# for i in range(trainX.shape[0]):
#     if label[0] != label[i] and k != label[i]:
#         eror_rate = comapre(img1, trainX[i])
#         print('Compare {} and {} error_rate: {}'.format(nameFace[0], nameFace[i], eror_rate))
#         k = label[i]


