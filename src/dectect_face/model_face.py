import glob
import os
import random
from shutil import copyfile
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

def build_model():
    name_home = 'E:/DataSet_CheckModel_Test/'
    subdirs = ['train/', 'val/']
    dataset_home = '../Image/'
    BASE_PATH = '../Image/DataIDCroped'
    BASE_PATH1 = 'E:/Image_DT/Fruit'
    all_file_peoples = glob.glob(BASE_PATH + '/*/*.jpg')
    all_file_people = []
    for i in all_file_peoples:
        all_file_people.append(i.replace('\\', '/'))
    all_file_others = glob.glob(BASE_PATH1 + '/*/*.jpg')
    all_file_other = []
    for i in all_file_others:
        all_file_other.append(i.replace('\\', '/'))

    # for i in all_file_other:
    #     i.replace('\\', '/')
    label_peple = []
    label_other = []

    for i in range(len(all_file_people)):
        label_peple.append(0)
    for i in range(len(all_file_others)):
        label_other.append(1)
    for sub in subdirs:
        labels = ['people/', 'other/']
        for label in labels:
            newdirs = name_home + sub + label
            os.makedirs(newdirs, exist_ok=True)

    # data = pd.read_csv("data_chekcmodel.csv")
    # img_data = data['path'].values
    # label = data['label'].values
    print(len(all_file_other))
    print(len(label_other))

    X_train_people, X_val_people, y_train_people, y_val_people = train_test_split(all_file_people, label_peple, train_size=0.8)
    X_train_other, X_val_other, y_train_other, y_val_other = train_test_split(all_file_other, label_other, train_size=0.8)
    # print(y_train_other)
    # return X_train, X_val, y_train, y_val

    # count = 0
    # for file in X_train_other:
    #     copyfile(BASE_PATH1 + '/' + file.split('/')[-2] + '/' + file.split('/')[-1], name_home + 'train/other/' + file.split('/')[-1])
    #     count += 1
    #     print(count)
    list(map(
        lambda file: copyfile(BASE_PATH + '/' + (file.replace('\\', '/')).split('/')[-2] + '/' + (file.replace('\\', '/')).split('/')[-1], name_home + 'train/people/' + str(random.randint(1 ,100)) + '_' + (file.replace('\\', '/')).split('/')[-1] ), X_train_people))
    list(map(lambda file: copyfile(BASE_PATH1 + '/' + (file.replace('\\', '/')).split('/')[-2] + '/' + (file.replace('\\', '/')).split('/')[-1], name_home + 'train/other/' + str(random.randint(1 ,100)) + '_' + str(random.randint(1 ,10000)) + '_' + str(random.randint(1 ,1000)) + '_' + (file.replace('\\', '/')).split('/')[-1]
                        ), X_train_other))
    print(len(X_train_other))

    list(map(lambda file: copyfile( BASE_PATH + '/' + (file.replace('\\', '/')).split('/')[-2] + '/' + (file.replace('\\', '/')).split('/')[-1], name_home + 'val/people/' + str(random.randint(1 ,100)) + '_' + (file.replace('\\', '/')).split('/')[-1] ),
             X_val_people))
    print(len(X_train_people))

    list(map(lambda file: copyfile( BASE_PATH1 + '/' + (file.replace('\\', '/')).split('/')[-2] + '/' + (file.replace('\\', '/')).split('/')[-1], name_home + 'val/other/'
                                    + str(random.randint(1, 100)) + '_' + str(random.randint(1 ,1000)) + '_' + str(random.randint(1 ,10000)) + '_' + (file.replace('\\', '/')).split('/')[-1]), X_val_other))
    print(len(X_val_other))

    Image_Width = 224
    Image_Height = 224
    Image_Channels = 3

    # training_data = data
    # print(training_data)


    print('done!!!')

build_model()
