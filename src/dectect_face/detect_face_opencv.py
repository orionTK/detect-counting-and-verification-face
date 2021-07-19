import cv2
import sys
import os
import argparse
from tqdm import tqdm
import math
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/DataSet', help='noi luu data can crop')
    parser.add_argument('--output_path', type=str, default='C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/DataCropedOpenCV', help='noi luu data da crop')

    args = parser.parse_args()
    return args

def crop_align_face(args):
    input_dir = args.input_path
    output_dir = args.output_path

    if not os.path.exists(input_dir):
        print('the input path is not exists!')
        sys.exit()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    count_no_find_face = 0
    count_crop_images = 0

    for root, dirs, files in tqdm(os.walk(input_dir)):

        # print(root)
        output_root = root.replace(input_dir, output_dir)
        if not os.path.exists(output_root):
            os.mkdir(output_root)

        for file_name in files:
            file_path = os.path.join(root, file_name)
            face_img = cv2.imread(file_path)

            if (face_img is None):
                print("Can't open image file")
                count_no_find_face += 1
                return 0

            print(face_img)
            print(face_img.shape)
            # scaleFactor = 1.05 và minNeighbors = 8
            #         minNeighbors=3,scaleFactor=1.3,minSize=(30, 30)
            faces = face_cascade.detectMultiScale(face_img, 1.05, 8, minSize=(100, 100))
            print('key: ', faces)
            # Kiểm tra ảnh có được detect
            if faces is None:
                print('%s do not find face' % file_path)
                count_no_find_face += 1
                continue

            height, width = face_img.shape[:2]

            for (x, y, w, h) in faces:
                r = max(w, h) / 2
                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int(r * 2)
                faceimg = face_img[ny:ny + nr, nx:nx + nr]
                face = cv2.resize(faceimg, (256, 256))
                count_crop_images += 1
                if file_name.split('.')[-1] == 'JPEG':
                    print('co jpeg')
                file_path_save = os.path.join(output_root, file_name)
                cv2.imwrite(file_path_save, face)

            count_crop_images += 1
    print('%d images crop successful!' % count_crop_images)
    print('%d images do not crop successful!' % count_no_find_face)
    print("rate: {}%", round(count_crop_images/(count_crop_images+count_no_find_face), 2))



if __name__ == '__main__':
    args = getArgs()
    crop_align_face(args)
    print("done!")