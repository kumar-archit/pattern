import numpy as np
import pylab as pl
import glob
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
def co(a,b,n):
    x0=0.0
    y0=0.0
    s=0.0
    for j in range(0,n):
        x0+=a[j]
        y0+=b[j]
    x0/=n
    y0/=n
    for j in range(0,n):
        s+=(a[j]-x0)*(b[j]-y0)
    s/=n
    return s

path = '/media/avi224/Local Disk/Sem5/CS669/rd_group13/*.txt'   
cov0=[[0,0],[0,0]]
cov=[]
cov.append([[0,0],[0,0]])
cov.append([[0,0],[0,0]])
cov.append([[0,0],[0,0]])
x=[]
files=glob.glob(path)
i=0
n=[0,0,0]   
for file in files:
    x.append([])
    f=open(file,'r')
    for line in f:
        c,d=line.split()
        x[i].append([])
        x[i].append([])
        x[i][0].append(float(c))
        x[i][1].append(float(d))
        n[i]+=1
    cov[i][0][1]=cov[i][1][0]=co(x[i][0],x[i][1],n[i])
    cov[i][0][0]=co(x[i][0],x[i][0],n[i])
    cov[i][1][1]=co(x[i][1],x[i][1],n[i])
    f.close()
    i+=1
X,Y=np.meshgrid(x[0][0],x[0][1])
plt.figure()
CS=plt.contour(X,Y,Z)
i=0
while i<3:
	pl.scatter(x[i][0],x[i][1])
	i+=1
#pl.show()
for i in range(0,3):
    for j in range(0,2):
        for k in range(0,2):
            cov0[j][k]+=cov[i][j][k]/3.0;
for i in range(0,3):
    for j in range(0,2):
        print(cov[i][j][0], cov[i][j][1])
    print() 
for i in range(0,2):
    print(cov0[i][0], cov0[i][1])