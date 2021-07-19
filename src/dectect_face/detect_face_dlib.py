import os
import sys
import dlib
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/DataIDFull', help='noi luu data can crop')
    parser.add_argument('--output_path', type=str, default='C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/DataCropedDlib', help='noi luu data da crop')

    args = parser.parse_args()
    return args

def crop_align_face(args):
    crop_width = 256
    simple_crop = False
    input_dir = args.input_path
    output_dir = args.output_path

    if not os.path.exists(input_dir):
        print('the input path is not exists!')
        sys.exit()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    face_detector = dlib.get_frontal_face_detector()
    print('key: ', face_detector)
    count_no_find_face = 0
    count_crop_images = 0

    for root, dirs, files in tqdm(os.walk(input_dir)):

        # print(root)
        output_root = root.replace(input_dir, output_dir)
        if not os.path.exists(output_root):
            os.mkdir(output_root)

        for file_name in files:
            file_path = os.path.join(root, file_name)
            img = cv2.imread(file_path)
            if img is None:
                continue
            detected_faces = face_detector(img, 1)
            # Kiểm tra ảnh có được detect
            if detected_faces is None:
                print('%s do not find face' % file_path)
                count_no_find_face += 1
                continue
            for i, face_rect in enumerate(detected_faces):

                height = face_rect.right() - face_rect.left()
                width = face_rect.bottom() - face_rect.top()

                print(height)
                print(width)
                if width >= crop_width and height >= crop_width:
                    image_to_crop = Image.open(file_path)

                    if simple_crop:
                        crop_area = (face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom())
                    else:
                        size_array = []
                        size_array.append(face_rect.top())
                        size_array.append(image_to_crop.height - face_rect.bottom())
                        size_array.append(face_rect.left())
                        size_array.append(image_to_crop.width - face_rect.right())
                        size_array.sort()
                        short_side = size_array[0]
                        crop_area = (face_rect.left() - size_array[0], face_rect.top() - size_array[0],
                                     face_rect.right() + size_array[0], face_rect.bottom() + size_array[0])

                    cropped_image = image_to_crop.crop(crop_area)
                    crop_size = (crop_width, crop_width)
                    cropped_image.thumbnail(crop_size)
                    file_path_save = os.path.join(output_root, file_name)
                    cropped_image.save(file_path_save)

            count_crop_images += 1
    print('%d images crop successful!' % count_crop_images)
    print('%d images do not crop successful!' % count_no_find_face)
    print("rate: {}%", round(count_crop_images/(count_crop_images+count_no_find_face), 2))



if __name__ == '__main__':
    # không nên sài

    args = getArgs()
    crop_align_face(args)
    print("done!")