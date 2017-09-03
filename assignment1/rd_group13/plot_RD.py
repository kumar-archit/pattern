import numpy as np
import pylab as pl
import glob   
path = '/media/avi224/Local Disk/Sem5/CS669/rd_group13/*.txt'   
x=[]
files=glob.glob(path)
i=0   
for file in files:
    x.append([])
    f=open(file,'r')
    for line in f:
    	#print(line)
    	c,d=line.split()
    	x[i].append([])
    	x[i].append([])
    	x[i][0].append(c)
    	x[i][1].append(d)
    f.close()
    i+=1
i=0
while i<3:
	pl.plot(x[i][0],x[i][1])
	i+=1
pl.show()