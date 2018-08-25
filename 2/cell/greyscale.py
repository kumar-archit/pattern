import glob
import PIL
from PIL import Image
files=glob.glob("2c/Test/*.png")
for file in files:
    im=Image.open(file)
    w,h=im.size
    pixel=im.load()
    all_pixel=[[0 for i in range(w)] for i in range(h)]
    for y in range(h):
        for x in range(w):
            cpixel=pixel[x,y]
            all_pixel[y][x]=cpixel
    sum=0
    sum0=0
    ssum=0
    ssum0=0
    for x in range(7):
        for y in range(7):
            sum0+=all_pixel[y][x]
            ssum0+=all_pixel[y][x]*all_pixel[y][x]
    file2 = file.split('.')[0]
    file2+=".txt"
    f=open(file2,"w")
    for i in range(h-6): #top left for each patch
        if i:
            for x in range(7):
                sum0+=all_pixel[i+6][x]-all_pixel[i-1][x]
                ssum0+=(all_pixel[i+6][x]-all_pixel[i-1][x])*(all_pixel[i+6][x]+all_pixel[i-1][x])
        sum=sum0
        ssum=ssum0
        mu=sum/49.0
        var=ssum/49.0-mu*mu
        f.write(str(mu)+" "+str(var)+"\n")


        for x in range(1,w-6):
            for y in range(i,i+7):
                sum+=all_pixel[y][x+6]-all_pixel[y][x-1]
                ssum+=(all_pixel[y][x+6]-all_pixel[y][x-1])*(all_pixel[y][x+6]+all_pixel[y][x-1])
            mu=sum/49.0
            var=ssum/49.0-mu*mu
            f.write(str(mu)+" "+str(var)+"\n")
    f.close()
