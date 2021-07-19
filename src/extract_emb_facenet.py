# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
from numpy import expand_dims
from keras.models import load_model
import glob, cv2
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
import timeit

# Lấy model facenet
model = load_model('./models/facenet_keras.h5')
print('Loaded Model')

def get_face(file_name, required_size=(160, 160)):
	file_name = file_name.replace('\\' or '//', '/')
	label = file_name.split('/')[-2]
	name = file_name.split('/')[-1]

	img = cv2.imread(file_name)  # đọc ra ảnh BGR
	img = cv2.resize(img, required_size, interpolation=cv2.INTER_AREA)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	face_array = asarray(img)

	return face_array,  label, name

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

def extract_feature_facenet(all_files):
	faces, names, labels = [], [], []

	for f in all_files:
		face, label, name = get_face(f)
		print("name", name)
		print("label", label)

		embedding = get_embedding(model, face)
		faces.append(embedding)
		names.append(name)
		labels.append(label)
	# savez_compressed('dataid_embding_from_facenet.npz', faces, labels, names)
	savez_compressed('lfw_embding_from_facenet.npz', faces, labels, names)

	return asarray(faces), asarray(labels), asarray(name)


# load a dataset that contains one subdir for each class that in turn contains images
# def load_dataset(directory):
# 	x, y, z = [], [], []
# 	j = 1
# 	# enumerate folders, on per class
# 	for subdir in listdir(directory):
# 		path = directory + subdir + '/'
#         #load
# 		faces= load_faces(path);
#         # create labels
# 		labelsT = [subdir for _ in range(len(faces))]
# 		labels = [j for _ in range(len(faces))]
# 		#summarize progress
# 		print('>Số lượng ảnh: %d : %s' % (len(faces), subdir))
# 		x.extend(faces)
# 		y.extend(labels)
# 		z.extend(labelsT)
# 		j += 1
# 	return asarray(x), asarray(y), asarray(z)
#
# testx, testy, nameFace = load_dataset('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/DataCroped/')
#save
# savez_compressed('./exc/dataset_compare_img_np.npz', testx, testy, nameFace)
#
#
#
# emd = list()
# for face_pixels in testx:
# 	embedding = get_embedding(model, face_pixels)
# 	emd.append(embedding)
# newTrainX = asarray(emd)
#
# #
# savez_compressed('./exc/dataset_compare_embeddings.npz', newTrainX, testy, nameFace)
#
# print(newTrainX)
# print(newTrainX.shape)
# print("done!")
if __name__ == "__main__":
	start_time = timeit.default_timer()
	# all_file_image = glob.glob('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src//Image/DataIDCroped/*/*.jpg')
	all_file_image = glob.glob('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src//Image/lfw-deepfunneled/lfw-deepfunneled/*/*.jpg')

	print(all_file_image[2].replace('\\', '/'))
	faces, names, labels = extract_feature_facenet(all_file_image)


	print(timeit.default_timer() - start_time)
	print('done!')