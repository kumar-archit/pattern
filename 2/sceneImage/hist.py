#to make color histogram
import PIL as pl
import glob
from PIL  import Image


def patch(c,r,all_pixel,file2):
    R=[0 for i in range(8)]
    B=[0 for i in range(8)]
    G=[0 for i in range(8)]
    H=[]

    file3 = file2.split('.')[0]
    file3+=".txt"
    f=open(file3,"a")

    for y_ in range(64):
        for x_ in range(64):
              R[int(all_pixel[r + y_][c + x_][0]/32)]+=1
              B[int(all_pixel[r + y_][c + x_][1]/32)]+=1
              G[int(all_pixel[r + y_][c + x_][2]/32)]+=1

    H= R+B+G
    for item in H:
     f.write(str(item)+" ")
    f.write("\n")
    f.close()

files=glob.glob("2b/*.jpg")
for file in files:
    im=Image.open(file)
    w,h=im.size
    perfect_x=(int)((w+63)/64)
    perfect_y=(int)((h+63)/64)
    pixel=im.load()
    all_pixel=[[[0,0,0] for i in range(perfect_x*64)] for i in range(perfect_y*64)]

    for y in range(h):
        for x in range(w):
            cpixel=pixel[x,y]
            all_pixel[y][x]=cpixel

    for py in range(perfect_y):
        for px in range(perfect_x):
            patch(px*64,py*64,all_pixel,file)
