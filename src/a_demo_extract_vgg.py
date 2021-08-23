from numpy import asarray
from os import listdir
from utils.vggface2 import *
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
def extract_feature_vvgface2(all_files):
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

    # if (i == 0):
    #     emb = vggface2.get_embeddings_vggface2_resnet50(samples)
    #     savez_compressed('lfw_embding_from_resnet50.npz', emb, names, labels)
    # else:
    #     if i == 1:
    #         emb = vggface2.get_embeddings_vggface2_setnet(samples)
    #         savez_compressed('lfw_embding_from_setnet.npz', emb, names, labels)
    #     else:
    #         emb = vggface2.get_embeddings_vggface2_vgg16(samples)
    #         savez_compressed('lfw_embding_from_vgg16.npz', emb, names, labels)

    emb = get_embeddings_vggface2_resnet50(samples)
    savez_compressed('demo_embding_from_resnet50.npz', emb, names, labels)

    emb1 = get_embeddings_vggface2_setnet(samples)
    savez_compressed('demo_embding_from_setnet.npz', emb1, names, labels)

    emb2 = get_embeddings_vggface2_vgg16(samples)
    savez_compressed('demo_embding_from_vgg16.npz', emb2, names, labels)

    return emb, names, labels

if __name__ == "__main__":
    start_time = timeit.default_timer()
    all_file_image = glob.glob('E:/Hoc/DoAn/src/Image/Demo/FaceVerification/*/*.jpg')

    emb, names, labels = extract_feature_vvgface2(all_file_image)

    print(timeit.default_timer() - start_time)
    print('done!')