import sys
import os.path
import csv

def CreateCsv(csvLine, csvLine1):
    BASE_PATH = csvLine
    BASE_PATH1 = csvLine1
    SEPARATOR = ","

    label = 0
    with open("img_label.csv", "w") as file:
        writer = csv.DictWriter(file, fieldnames=('path', 'label'))
        writer.writeheader()
        for dirname, dirnames, filenames in os.walk(BASE_PATH):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                print(subject_path)
                for filename in os.listdir(subject_path):
                    print(filename)
                    abs_path = ("%s/%s" % (subject_path, filename))
                    print(abs_path)
                    writer.writerow({'path': abs_path,'label': label})
        label = label + 1

        for dirname, dirnames, filenames in os.walk(BASE_PATH1):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    abs_path = "%s/%s" % (subject_path, filename)
                    writer.writerow({'path': abs_path,'label': label})

if __name__ == '__main__':
    # mặt người label 0, vật khác label 1
    CreateCsv('../Image/UTKCropped', '../Image/FruitDataset')

    print('done')