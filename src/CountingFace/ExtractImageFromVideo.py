import cv2
import glob
import os
def FrameCapture(pathVideo, pathImage):
    vidObj = cv2.VideoCapture(pathVideo)
    nameFolder = pathVideo.split("\\")[-1].split(".")[0]
    nameImage = pathVideo.split("\\")[-1].split(".")[-1]
    subFolder = pathImage + "\\" + nameFolder
    print(pathVideo)
    print(subFolder)
    if os.path.exists(subFolder) != True:
        os.mkdir(subFolder)

    count = 0
    count_name = 0
    success = 1

    while success:
        success, image = vidObj.read()
        nameSubImage = pathImage + "\\" + nameFolder + "\\" + nameImage + "_" + str(count_name) + ".jpg"
        if count % 15 == 0:
            cv2.imwrite(nameSubImage, image)
            count_name += 1
        count += 1
    print(count)

def ExtractImageFromVideo(pathVideo, pathImage):
    listVideos = glob.glob(r"F:\Secret\Video\*.*")
    for i, nameVideo in enumerate(listVideos):
        FrameCapture(nameVideo, pathImage)
    
if __name__ == '__main__':
    # Calling the function
    pathVideo = r'F:\Secret\Video'
    pathImage=r"F:\Secret\ImageFromVideo"
    ExtractImageFromVideo(pathVideo, pathImage)