import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import glob
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import math
avg=[]
cov0=[[0,0],[0,0]]
cov=[]
cinv=[]
test=[]
x=[]
W=[]
w=[]
n=[0,0,0]
col=["red","blue","green"]
k=0
tot=0
def fn(X,k):
    return -0.5*math.log(np.linalg.det(cinv[k]))+math.log(n[k]*1.0/tot)-0.5*np.dot(np.dot(np.transpose(np.subtract(X,avg[k])),cinv[k]),np.subtract(X,avg[k]))
def g(i,j,cov1,cov2):
    ci=np.linalg.inv(cov1)
    cj=np.linalg.inv(cov2)
    W=np.subtract(cj,ci)
    W*=0.5
    w=np.subtract(np.dot(ci,avg[i]),np.dot(cj,avg[j]))
    w=np.transpose(w)
    w0=-0.5*math.log(np.linalg.det(cov1)/np.linalg.det(cov2))
    w0-=0.5*np.dot(np.dot(np.transpose(avg[i]),ci),avg[i])
    w0+=0.5*np.dot(np.dot(np.transpose(avg[j]),cj),avg[j])
    w0+=math.log(n[i]/n[j])
    #print(W, w, w0)
    X = np.linspace(-15,15)
    Y = np.linspace(-15,15)[:, None]
    plt.contour(X,Y.ravel(),W[0][0]*X*X+W[1][1]*Y*Y+(W[1][0]+W[0][1])*X*Y+w[0]*X+w[1]*Y+w0,[0])
def co(a,b,n):
    x0=0.0
    y0=0.0
    s=0.0
    for j in range(0,n):
        x0+=a[j]
        y0+=b[j]
    x0/=n
    y0/=n
    avg.append([x0,y0])
    for j in range(0,n):
        s+=(a[j]-x0)*(b[j]-y0)
    s/=n
    return s
path = '/media/avi224/Local Disk/Sem5/CS669/pattern/assignment1/solution2/Train/*.txt'   
cov.append([[0,0],[0,0]])
cov.append([[0,0],[0,0]])
files=glob.glob(path)
i=0  
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
    tot+=n[i]
    cov[i][0][1]=cov[i][1][0]=co(x[i][0],x[i][1],n[i])
    cov[i][0][0]=co(x[i][0],x[i][0],n[i])
    cov[i][1][1]=co(x[i][1],x[i][1],n[i])
    f.close()
    i+=1
for i in range(0,2):
    pl.scatter(x[i][0],x[i][1])
for i in range(0,2):
    cinv.append(np.linalg.inv(cov[i]))
for j in range(0,2):
    for k in range(0,2):
        for i in range(0,2):
            cov0[j][k]+=cov[i][j][k]
        cov0[j][k]/=2.0
for i in range(0,2):
    #g(i,(i+1)%2,cov[i],cov[(i+1)%2])
    g(i,(i+1)%2,cov0,cov0)
path = '/media/avi224/Local Disk/Sem5/CS669/pattern/assignment1/solution2/Test/*.txt'
i=0
for file in files: 
    f=open(file,'r')
    test.append([])
    for line in f:
        c,d=line.split()
        test[i].append([float(c),float(d)])
    i+=1
cnt=[]
for i in range(0,2):
    cnt.append([0,0,0])
to=0
for i in range(0,2):
    correct=0
    for j in test[i]:
        to+=1
        max=0
        maxi=-100000000000
        for k in range(0,2):
            if fn(j,k)>maxi:
                max=k
                maxi=fn(j,k)
        for k in range(0,2):
            if fn(j,k)==maxi:
                cnt[i][k]+=1
    correct+=cnt[i][i]
    print(cnt[i][0], cnt[i][1], cnt[i][2])
print("classification accuracy:",correct/to)
for i in range(0,2):
    to=0
    to1=0
    for j in range(0,2):
        to+=cnt[j][i]
        to1+=cnt[i][j]
    print("precision of class",i+1,":",cnt[i][i]/to)
    print("recall of class",i+1,":",cnt[i][i]/to1)
pl.show()