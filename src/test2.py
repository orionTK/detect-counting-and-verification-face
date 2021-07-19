import os
import glob
import cv2
import mxnet as mx
from align_mtcnn.mtcnn_detector import MtcnnDetector
class CountingFace:
    def Read_File(self):
        with open("C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/label_count.txt", 'r') as f:
            labels = f.readlines()

        print((labels[298].split(" ")));
        return
    def Counting(self):
        true_positive = 0.0
        true_negative = 0.0
        total = 0.0

        with open("C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/label_count.txt", 'r') as f:
            labels = f.readlines()
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
        #
        gpu = -1
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)
        mtcnn = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)
        # filenames = glob.glob("C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/Count/*.jpg")
        dir = "C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/Count/";

        for labels in labels:
            if len(labels.split(" ")) > 1:
                filename = dir + labels.split(" ")[0]
                print(filename)
                print((labels.split(" ")[-1]).split("\n"))
                # number_face = int((labels.split(" ")[-1]).split("\n")[0])
                number_face = int((labels.split(" ")[1]).split("\n")[0])
                face_img = cv2.imread(filename)
                ret = mtcnn.detect_face(face_img)
                if ret is None:
                    continue
                bbox, points = ret
                if bbox is None:
                    continue
                print(bbox)
                true_positive += bbox.shape[0];
                total += number_face
                print(labels.split(" ")[-1].split("\n")[0])
                print("shape ", bbox.shape[0])
                print("numbe ", number_face)

                print(true_positive)
        return true_positive / total
if __name__ == '__main__':
    print(CountingFace.Counting(""))