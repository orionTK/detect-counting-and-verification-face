import glob
import os
def RenameVideo(pathVideo):
    listVideos = glob.glob("F:\Secret\Video\*.*")
    print(listVideos)
    for i, nameVideo in enumerate(listVideos):
        newPathVideo = pathVideo + "\KVideo" + str(i) + "." + nameVideo.split(".")[-1]
        os.rename(nameVideo,newPathVideo )

if __name__ == '__main__':
    RenameVideo('F:\Secret\Video')