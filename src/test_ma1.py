
import numpy as np
from numpy import savez_compressed, load

def findManhattanDistance(a, b):
	return sum(abs(x-y) for x, y in zip(a,b))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def is_match(known_embedding, candidate_embedding, thresh=0.4):
    # Calculate cosine
    # score = cosine(known_embedding, candidate_embedding)

    # calculate
    # EuclideanDistance = findEuclideanDistance(known_embedding, candidate_embedding)
    EuclideanDistance = findEuclideanDistance(known_embedding, candidate_embedding)

    score_ma = findManhattanDistance(known_embedding, candidate_embedding)
    print("<----------score: ", score_ma)
    if score_ma <= thresh:
        return True
    return False

def acc(i):
    true_positive = 0.0
    true_negative = 0.0
    total = 0.0

    true_positive1 = 0.0
    true_negative1 = 0.0
    total1 = 0.0

    true_positive2 = 0.0
    true_negative2 = 0.0
    total2 = 0.0

    true_positive3 = 0.0
    true_negative3 = 0.0
    total3 = 0.0
    [positives, negatives, false_positives, false_negatives] = [0., 0., 0., 0.]
    [positives1, negatives1, false_positives1, false_negatives1] = [0., 0., 0., 0.]
    [positives2, negatives2, false_positives2, false_negatives2] = [0., 0., 0., 0.]
    [positives3, negatives3, false_positives3, false_negatives3] = [0., 0., 0., 0.]


    data = load('dataid_embding_from_facenet.npz')
    data1 = load('dataid_embding_from_resnet50.npz')
    data2 = load('dataid_embding_from_setnet.npz')
    data3 = load('dataid_embding_from_vgg16.npz')
    # if (i == 0):
    #     # data = load('dataid_embding_from_resnet50.npz')
    #     data = load('dataid_embding_from_facenet.npz')
    #
    # else:
    #
    #     if i == 1:
    #         data = load('dataid_embding_from_setnet.npz')
    #     else:
    #         data = load('dataid_embding_from_vgg16.npz')

    emb = data['arr_0']
    labels = data['arr_1']
    names = data['arr_2']

    emb1 = data1['arr_0']
    labels1 = data1['arr_1']
    names1 = data1['arr_2']

    emb2 = data2['arr_0']
    labels2 = data2['arr_1']
    names2 = data2['arr_2']

    emb3 = data3['arr_0']
    labels3 = data3['arr_1']
    names3 = data3['arr_2']

    for i in range(len(emb) - 1):
        for j in range(i + 1, len(emb)):
            print("i ", i)
            print("j ", j)
            if labels[i]!= labels[j]:
                if is_match(emb[i], emb[j]):
                    false_negatives += 1
                else:
                    true_negative  += 1
                    print(true_negative)
                negatives += 1
            else:
                print("================", names[i], " === ", names[j])
                print("================", labels[i], " === ", labels[j])
                if is_match(emb[i], emb[j]) == False:
                    false_positives += 1
                else:
                    true_positive += 1
                positives += 1
            total += 1
            print('total: ', total)


            #1
            if labels1[i]!= labels1[j]:
                if is_match(emb1[i], emb1[j]):
                    false_negatives1 += 1
                else:
                    true_negative1  += 1
                    print(true_negative1)
                negatives1 += 1
            else:
                print("================", names1[i], " === ", names1[j])
                print("================", labels1[i], " === ", labels1[j])
                if is_match(emb1[i], emb1[j]) == False:
                    false_positives1 += 1
                else:
                    true_positive1 += 1
                positives1 += 1

            total1 += 1
            print('total: ', total1)

            #2
            if labels2[i] != labels2[j]:
                if is_match(emb2[i], emb2[j]):
                    false_negatives2 += 1
                else:
                    true_negative2 += 1
                    print(true_negative2)
                negatives2 += 1
            else:
                print("================", names2[i], " === ", names2[j])
                print("================", labels2[i], " === ", labels2[j])
                if is_match(emb2[i], emb2[j]) == False:
                    false_positives2 += 1
                else:
                    true_positive2 += 1
                positives2 += 1

            total2 += 1
            print('total: ', total2)

            #3
            if labels3[i] != labels3[j]:
                if is_match(emb3[i], emb3[j]):
                    false_negatives3 += 1
                else:
                    true_negative3 += 1
                    print(true_negative3)
                negatives3 += 1
            else:
                print("================", names3[i], " === ", names3[j])
                print("================", labels3[i], " === ", labels3[j])
                if is_match(emb3[i], emb3[j]) == False:
                    false_positives3 += 1
                else:
                    true_positive3 += 1
                positives3 += 1

            total3 += 1
            print('total: ', total3)

    if negatives != 0:
        far = false_positives / negatives
    else:
        far = 0
    if positives != 0:
        frr = false_negatives / positives
    else:
        frr = 0


    return (true_negative + true_positive) / total, (true_negative1 + true_positive1) / total1, (true_negative2 + true_positive2) / total2, (true_negative3 + true_positive3) / total3


if __name__ == "__main__":

    # emb, names, labels = extract_feature_vvgface2(all_file_image, 2)
    acc0, acc1, acc2, acc3 = acc(0)
    print("acc_se = ", acc0)
    print("acc_fe = ", acc1)
    print("acc_re = ", acc2)
    print("acc_vgg = ", acc3)