from numpy import asarray
from os import listdir
from utils import vggface2
import timeit
from PIL import Image
from scipy.spatial.distance import cosine
from keras.preprocessing import image
from numpy import savez_compressed
import glob, cv2
import numpy as np
from numpy import savez_compressed, load


def get_face(file_name, required_size=(224, 224)):
    file_name = file_name.replace('\\' or '//', '/')
    name  = file_name.split('/')[-2]
    label = file_name.split('/')[-1]
    print(file_name.split('/')[-2])
    image = cv2.imread(file_name)
    image = Image.fromarray(image)
    img_data = image.resize(required_size)
    face_array = asarray(img_data)
    return face_array, name, label



def get_img(img):
    print(img)
    img = image.load_img(img, target_size=(224, 224))
    img_data = image.img_to_array(img)
    face_array = asarray(img_data)
    return face_array

# cũ
def extract_feature(path, i):
    facesTest, nameTest = [], []
    for filename in listdir(path):
        img_path = path + filename
        if filename.find('.jpg') != -1:
            face = get_img(img_path)
            if (i == 0):
                emb = vggface2.get_embeddings_vggface2_resnet50(face)
            else:
                if i == 1:
                    emb = vggface2.get_embeddings_vggface2_setnet(face)
                else:
                    emb = vggface2.get_embeddings_vggface2_vgg16(face)
            facesTest.append(emb)
            nameTest.append(filename.split("/")[-1])
    return facesTest, nameTest

def get_embeddings(path, i):
    facesTest, nameTest = [], []
    for filename in listdir(path):
        img_path = path + filename
        if filename.find('.jpg') != -1:
            face = get_img(img_path)
            if (i == 0):
                emb = vggface2.get_embeddings_vggface2_resnet50(face)
            else:
                if i == 1:
                    emb = vggface2.get_embeddings_vggface2_setnet(face)
                else:
                    emb = vggface2.get_embeddings_vggface2_vgg16(face)
            facesTest.append(emb)
            nameTest.append(filename.split("/")[-1])
    return facesTest, nameTest



# 28/2/2021 use
def extract_feature_vvgface2(all_files, i):
    faces, names, labels = [], [], []
    for f in all_files:
        face, name, label = get_face(f)
        print("name", name)
        print("label", label)
        faces.append(face)
        names.append(name)
        labels.append(label)


    samples = asarray(faces, 'float32')
    print(len(samples))
    if (i == 0):
        emb = vggface2.get_embeddings_vggface2_resnet50(samples)
        savez_compressed('lfw_embding_from_resnet50.npz', emb, names, labels)
    else:
        if i == 1:
            emb = vggface2.get_embeddings_vggface2_setnet(samples)
            savez_compressed('lfw_embding_from_setnet.npz', emb, names, labels)
        else:
            emb = vggface2.get_eembding_from_vgg16.npz', emb, names, labels)

    # code cũmbeddings_vggface2_vgg16(samples)
    #             savez_compressed('lfw_
    # x = []
    # y = []
    # z = []
    # j = 1
    # # enumerate folders, on per class
    # for subdir in listdir(directory):
    #     path = directory + subdir + '/'
    #     # load
    #
    #     faces, nameTest = extract_feature(path, i);
    #     # create labels
    #     labels = [j for _ in range(len(faces))]
    #     # summarize progress
    #     print('>Số lượng ảnh: %d : %s' % (len(faces), subdir))
    #     x.extend(faces)
    #     y.extend(labels)
    #     z.extend(nameTest)
    #     j += 1
    # return asarray(x), asarray(y), asarray(z)
    # np.savetxt('dataid_embdingfrom_vggface2.csv', emb, delimiter=',', fmt='%d')
    # np.savetxt('dataid_name_vggface2.csv', names, delimiter=',', fmt='%d')
    # np.savetxt('dataid_labels_vggface2.csv', labels, delimiter=',', fmt='%d')


    return emb, names, labels


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
    start_time = timeit.default_timer()
    # all_file_image = glob.glob('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src//Image/DataIDCroped/*/*.jpg')
    all_file_image = glob.glob('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src//Image/lfw-deepfunneled/lfw-deepfunneled/*/*.jpg')

    print(all_file_image[2].replace('\\', '/'))
    for i in range(0, 3):
        emb, names, labels = extract_feature_vvgface2(all_file_image, i)

    # chạy file acc_verfication... để tìm acc

    # acc1 = acc(1)
    # acc2 = acc(2)
    #
    # print('acc1 = ', acc1)
    # print('acc2 = ', acc2)

    # # kiến trúc resnet50
    # feature_train, label_train, name_test = extract_feature_vvgface2('./DataCroped/', 0)
    # print(feature_train.shape)
    #
    # savez_compressed('./exc/veactor_feature_vggface2_resnet50.npz', feature_train, label_train, name_test)
    #
    # # KIẾN trúc senet50
    # feature_train, label_train, name_test = extract_feature_vvgface2('./DataCroped/', 1)
    # print(feature_train.shape)
    #
    # savez_compressed('./exc/veactor_feature_vggface2_senet50.npz', feature_train, label_train, name_test)
    #
    # # KIẾN trúc vgg16
    # feature_train, label_train, name_test = extract_feature_vvgface2('./DataCroped/', 2)
    # print(feature_train.shape)
    #
    # savez_compressed('./exc/veactor_feature_vggface2_vgg16.npz', feature_train, label_train, name_test)

    print(timeit.default_timer() - start_time)
    print('done!')