from numpy import asarray
from os import listdir
from ...src.utils import vggface2
import timeit
from keras.preprocessing import image
from numpy import savez_compressed
import cv2

def extract_face(img, required_size=(224, 224)):
    print(img)
    pixels = cv2.imread(img)
    img_data = image.resize(required_size)
    face_array = asarray(img_data)
    return face_array

def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]

	samples = asarray(faces, 'float32')

	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat
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


def extract_feature_vvgface2(directory, i):
    x = []
    y = []
    z = []
    j = 1
    # enumerate folders, on per class
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        # load

        faces, nameTest = extract_feature(path, i);
        # create labels
        labels = [j for _ in range(len(faces))]
        # summarize progress
        print('>Số lượng ảnh: %d : %s' % (len(faces), subdir))
        x.extend(faces)
        y.extend(labels)
        z.extend(nameTest)
        j += 1
    return asarray(x), asarray(y), asarray(z)

if __name__ == "__main__":
    start_time = timeit.default_timer()
    # kiến trúc resnet50
    feature_train, label_train, name_test = extract_feature_vvgface2('./DataCroped/', 0)
    print(feature_train.shape)

    savez_compressed('./exc/veactor_feature_vggface2_resnet50.npz', feature_train, label_train, name_test)

    # KIẾN trúc senet50
    feature_train, label_train, name_test = extract_feature_vvgface2('./DataCroped/', 1)
    print(feature_train.shape)

    savez_compressed('./exc/veactor_feature_vggface2_senet50.npz', feature_train, label_train, name_test)

    # KIẾN trúc vgg16
    feature_train, label_train, name_test = extract_feature_vvgface2('./DataCroped/', 2)
    print(feature_train.shape)

    savez_compressed('./exc/veactor_feature_vggface2_vgg16.npz', feature_train, label_train, name_test)

    print(timeit.default_timer() - start_time)
    print('done!')