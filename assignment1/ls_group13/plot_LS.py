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
x=[]
W=[]
w=[]
n=[0,0,0]
test=[]
tot=0
col=["ro","bo","go"]
k=0
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
    X = np.linspace(-10,25)
    Y = np.linspace(-15,20)[:, None]
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
path = '/media/avi224/Local Disk/Sem5/CS669/pattern/assignment1/ls_group13/Train/*.txt'   
cov.append([[0,0],[0,0]])
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
for i in range(0,3):
    pl.plot(x[i][0],x[i][1],col[i])
for i in range(0,3):
    cinv.append(np.linalg.inv(cov[i]))
for j in range(0,2):
    for k in range(0,2):
        for i in range(0,3):
            cov0[j][k]+=cov[i][j][k]
        cov0[j][k]/=3.0
for i in range(0,3):
    #g(i,(i+1)%3,cov[i],cov[(i+1)%3])
    g(i,(i+1)%3,cov0,cov0)
path = '/media/avi224/Local Disk/Sem5/CS669/pattern/assignment1/ls_group13/Test/*.txt'
files=glob.glob(path)
i=0
for file in files: 
    f=open(file,'r')
    test.append([])
    for line in f:
        c,d=line.split()
        test[i].append([float(c),float(d)])
    i+=1
cnt=[]
for i in range(0,3):
    cnt.append([0,0,0])
to=0
for i in range(0,3):
    correct=0
    for j in test[i]:
        to+=1
        max=0
        maxi=-100000000000
        for k in range(0,3):
            if fn(j,k)>maxi:
                max=k
                maxi=fn(j,k)
        for k in range(0,3):
            if fn(j,k)==maxi:
                cnt[i][k]+=1
    correct+=cnt[i][i]
    print(cnt[i][0], cnt[i][1], cnt[i][2])
print("classification accuracy:",correct/to)
pr=0
rec=0
fm=0
for i in range(0,3):
    to=0
    to1=0
    for j in range(0,3):
        to+=cnt[j][i]
        to1+=cnt[i][j]
    print("Precision of class",i+1,"=",cnt[i][i]/to)
    pr+=cnt[i][i]/to
    print("Recall of class",i+1,"=",cnt[i][i]/to1)
    rec+=cnt[i][i]/to
    print("F-Measure of class",i+1,"=",2*cnt[i][i]*cnt[i][i]/(to*to1))
    fm+=2*cnt[i][i]*cnt[i][i]/(to*to1)
    print()
print("Mean precision=",pr/3)
print("Mean recall=",rec/3)
print("Mean F-Measure=",fm/3)
pl.show()