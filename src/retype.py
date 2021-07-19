from numpy import asarray
from os import listdir
import timeit
from PIL import Image
from scipy.spatial.distance import cosine
from numpy import savez_compressed
import glob, cv2
import numpy as np
from numpy import savez_compressed, load


def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # Calculate cosine
    score = cosine(known_embedding, candidate_embedding)

    # calculate L2
    point1 = np.array(known_embedding)
    point2 = np.array(candidate_embedding)
    ddistance = np.linalg.norm(point1 - point2)

    print("<----------score: ", score)
    if score <= thresh:
        return True
    return False

def acc(i):
    true_positive = 0.0
    true_negative = 0.0
    total = 0.0
    [positives, negatives, false_positives, false_negatives] = [0., 0., 0., 0.]
    if (i == 0):
        data = load('dataid_embding_from_resnet50.npz')
    else:
        if i == 1:
            data = load('dataid_embding_from_setnet.npz')
        else:
            data = load('dataid_embding_from_vgg16.npz')

    emb = data['arr_0']
    labels = data['arr_1']
    names = data['arr_2']
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

    if negatives != 0:
        far = false_positives / negatives
    else:
        far = 0
    if positives != 0:
        frr = false_negatives / positives
    else:
        frr = 0


    return (true_negative + true_positive) / total


if __name__ == "__main__":

    # emb, names, labels = extract_feature_vvgface2(all_file_image, 2)
    acc1 = acc(1)
    acc2 = acc(2)
    print("acc1 = ", acc1)
    print("acc2 = ", acc2)