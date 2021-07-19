# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir

import cv2
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray

def extract_img(filename):
    img = cv2.imread(filename)  # đọc ra ảnh BGR
    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_array = asarray(img)
    return face_array


def load_faces(directory,i):
    facesTest , facesTrain, nameTest, nameTrain= [], [], [], []

    # dem=0;
    k = 0;
    # enumerate files
    for filename in listdir(directory):
        if filename.find('.jpg') != -1:
            path = directory + filename
            face = extract_img(path)
            if filename.find('_0ID.jpg') != -1:
                facesTrain.append(face)
                nameTrain.append(filename)
            else:
                facesTest.append(face)
                nameTest.append(filename)


	if i == 0:
		return facesTrain
	else:
		return facesTest


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory, i):
    x = []
    y = []
    j = 1
	# enumerate folders, on per class
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        #load
        faces= load_faces(path, i);
        # create labels
        if i == 0:
            labels = [j]

        else:
            labels = [j for _ in range(len(faces))]
        #summarize progress
        print('>Số lượng ảnh: %d : %s' % (len(faces), subdir))
        x.extend(faces)
        y.extend(labels)
        j += 1
    return asarray(x), asarray(y)

# load train dataset
trainx, trainy = load_dataset('C:/Users/SV_Guest/Desktop/Kieu/kieu/src/Data/', 0)

# # load test dataset
testx, testy = load_dataset('C:/Users/SV_Guest/Desktop/Kieu/kieu/src/Data/', 1)
print("train")
print(trainx.shape, trainy)
print("test")
print(testx.shape, testy)

# # save arrays to one file in compressed format
savez_compressed('./exc/dataf.npz', trainx, trainy, testx, testy)


