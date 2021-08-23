
from scipy.spatial.distance import cosine
import numpy as np
from numpy import savez_compressed, load
from multiprocessing.pool import ThreadPool
import cv2
pool = ThreadPool(processes=1)
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
import xlsxwriter


def findManhattanDistance(a, b):
	return sum(abs(x-y) for x, y in zip(a,b))

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findEuclideanDistance(source_representation, test_representation):
    source_representation = l2_normalize(source_representation)
    test_representation = l2_normalize(test_representation)
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def is_match(known_embedding, candidate_embedding, thresh):
    scores = []
    # Calculate cosine
    score = cosine(known_embedding, candidate_embedding)

    # calculate
    # EuclideanDistance = findEuclideanDistance(known_embedding, candidate_embedding)
    EuclideanDistance = findEuclideanDistance(known_embedding, candidate_embedding)
    scores.append(score)


    print("<----------distance: ", score, "\n\n")
    if score <= thresh:
        return True, score

    return False, score

def count_value(data, thresh, method):
    path = "E:/Hoc/DoAn/src/Image/Demo/FaceVerification"
    imgs1 = []
    imgs2 = []
    check_imgs = []
    check_pred = []
    scores = []
    true_positive = 0.0
    true_negative = 0.0
    total = 0.0
    emb = data['arr_0']
    labels = data['arr_1']
    names = data['arr_2']
    for i in range(len(emb) - 1):

        for j in range(i + 1, len(emb)):
            file_name = path + '/' + labels[i] + '/' + names[i]
            imgs1.append(file_name)
            # imgs1.append(mpimg.imread(file_name))
            print("================", names[i], " === ", names[j])
            print("================", labels[i], " === ", labels[j])
            check, score = is_match(emb[i], emb[j],thresh)
            scores.append(score)
            if (check == True):
                check_pred.append('Cung mot nguoi')
            else:
                check_pred.append('Khong cung mot nguoi')
            file_name = path + '/' + labels[j] + '/' + names[j]
            imgs2.append(file_name)

            # imgs2.append(mpimg.imread(file_name))
            if labels[i]!= labels[j] and  check == False:
                    true_negative  += 1
                    print(true_negative)
                    check_imgs.append('correct')


            else:

                if check == True and labels[i]== labels[j]:
                    true_positive += 1
                    check_imgs.append('correct')
                else:
                    check_imgs.append('incorrect')

            total += 1
    plt.figure(figsize=(10,9))
    print(len(check_imgs))
    name_fileexl = 'result_verification_' + method + '.xlsx'
    workbook = xlsxwriter.Workbook(name_fileexl)
    worksheet = workbook.add_worksheet()
    i = 0
    worksheet.write('A1', 'Image 1')
    worksheet.write('F1', 'Image 2')
    worksheet.write('K1', 'Distance')
    worksheet.write('L1', 'Predict')

    for n in range(len(check_imgs)):
        name_img1 = 'A' + str(n + i * 13 + 2)
        name_img2 = 'F' + str(n + i * 13 + 2)
        name_dis = 'K' + str(n + i * 13 + 5)
        name_pred = 'L' + str(n + i * 13 + 5)
        name_result = 'M' + str(n + i * 13 + 5)
        i += 1

        worksheet.insert_image(name_img1, imgs1[n])
        worksheet.insert_image(name_img2, imgs2[n])
        worksheet.write(name_dis, scores[n])
        worksheet.write(name_pred, check_pred[n])
        worksheet.write(name_result, check_imgs[n])
    workbook.close()
    # for n in range(len(check_imgs)):
    #     plt.subplot(len(check_imgs)//2,2, n+1)
    #     plt.imshow(imgs1[n])
    #     plt.imshow(imgs2[n])
    #     color = 'blue'
    #     if check_imgs[n] == True:
    #         color = 'blue'
    #     title = 'dis = ' + str(math.ceil(scores[n]*10000)/100000)
    #     plt.title(title, color=color)
    #     plt.axis('off')
    #     _ = plt.suptitle('blue: correct, red: incorrect')
    # plt.show()
    return true_negative, true_positive, total

def acc():

    data_facenet = load('./Image/Demo/FaceVerification/demo_embding_from_facenet.npz')
    data_resnet = load('./Image/Demo/FaceVerification/demo_embding_from_resnet50.npz')
    data_setnet = load('./Image/Demo/FaceVerification/demo_embding_from_setnet.npz', 'setnet')
    data_vgg16 = load('./Image/Demo/FaceVerification/demo_embding_from_vgg16.npz', 'vgg16')


    true_negative, true_positive, total = count_value(data_facenet, thresh = 0.5, method='facenet')
    true_negative1, true_positive1, total1 = count_value(data_resnet, thresh = 0.5, method= 'resnet50')
    true_negative2, true_positive2, total2 = count_value(data_setnet, thresh = 0.5, method='setnet')
    true_negative3, true_positive3, total3 = count_value(data_vgg16, thresh = 0.4, method='vgg16')


    return (true_negative + true_positive) / total, (true_negative1 + true_positive1) / total1, (true_negative2 + true_positive2) / total2, (true_negative3 + true_positive3) / total3


if __name__ == "__main__":

    acc_facenet, acc_resnet, acc_setnet, acc_vgg16 = acc()
    print("acc_facenet = ", math.ceil(acc_facenet*1000000)/1000000)
    print("acc_resnet = ", math.ceil(acc_resnet*10000)/10000)
    print("acc_setnet = ", math.ceil(acc_setnet*10000)/10000)
    print("acc_setnet = ", math.ceil(acc_vgg16*10000)/10000)