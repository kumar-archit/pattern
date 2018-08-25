import numpy as np
import glob
import sys
import math
import matplotlib.pyplot as plt
###############################################
pathTrain='1a/Train/Class?.txt'#stored as x
pathTest='1a/Test/Class?.txt'#stored as y
noOfClasses=3
dim=2
eta=0.2
eta_diff=0.2
noOfEtaIter=5
noOfPercIter=5
###############################################

def inp():
    global noOfClasses
    global pathTrain
    x=[]
    y=[]

    files=glob.glob(pathTrain)
    j=0
    for file in sorted(files):
        x.append([])
        f=open(file,'r')
        for line in f:
            c=[float(w) for w in line.split()]
            c.insert(0,1)
            x[j].append(c)
        f.close()
        j+=1

    files=glob.glob(pathTest)
    j=0
    for file in sorted(files):
        y.append([])
        f=open(file,'r')
        for line in f:
            c=[float(w) for w in line.split()]
            c.insert(0,1)
            y[j].append(c)
        f.close()
        j+=1
    a=np.array(x)
    b=np.array(y)
    meanx=[]
    meany=[]
    for i in range(noOfClasses):
        meanx.append((np.mean(a[i],axis=0)).tolist())
    for i in range(noOfClasses):
        meany.append((np.mean(b[i],axis=0)).tolist())

    return x,y

def isNotEmptyD_m(x0,x1,a):
    count=0
    for p in x0:
        val=np.asscalar(np.matmul(a,p))
        if val < 0:
            count+=1

    for p in x1:
        val=np.asscalar(np.matmul(a,p))
        if val > 0:
            count+=1
    if count>0:
        return 1
    else:
        return 0

def deltaA(x0,x1,a):
    global dim
    sum=[0 for i in range(dim +1)]
    for p in x0:
        val=np.asscalar(np.matmul(a,p))
        if val < 0:
            sum= np.add(sum,np.multiply(val,p).tolist()).tolist()

    for p in x1:
        val=np.asscalar(np.matmul(a,p))
        if val > 0:
            sum=np.add(sum,np.multiply(val,p).tolist()).tolist()
    return sum
def deltaA2(x0,x1,a):
    global dim
    sum=[0 for i in range(dim +1)]
    for p in x0:
        val=np.asscalar(np.matmul(a,p))
        if val < 0:
            sum=  np.add(sum,np.multiply(1,p).tolist()).tolist()

    for p in x1:
        val=np.asscalar(np.matmul(a,p))
        if val > 0:
            sum=np.add(sum,np.multiply(-1,p).tolist()).tolist()
    return sum

def deltaA3(x0,x1,a):
    global dim
    sum=[0 for i in range(dim +1)]
    for p in x0:
        val=np.asscalar(np.matmul(a,p))
        if val < 0:
            Norm = np.linalg.norm(p)
            Norm=np.power(Norm,2)
            val/=Norm
            sum= np.add(sum,np.multiply(val,p).tolist()).tolist()

    for p in x1:
        val=np.asscalar(np.matmul(a,p))
        if val > 0:
            Norm = np.linalg.norm(p)
            Norm=np.power(Norm,2)
            val/=Norm
            sum=np.add(sum,np.multiply(val,p).tolist()).tolist()
    return sum


def perceptron(x0,x1,a,eta):
    global noOfPercIter
    sign_eta=1
    j=0
    while(isNotEmptyD_m(x0,x1,a) ):
        sum=deltaA2(x0,x1,a)
        sum=np.multiply(sign_eta*eta, sum).tolist()
        a=np.add(a,sum).tolist()
    return a

def getConfMat(x,y,eta):
    global noOfClasses
    confM= [ [0 for ii in range(noOfClasses)] for jj in range(noOfClasses)]
    a=[[1 for p in range(dim+1)] for q in range(noOfClasses)]
    for j in range(noOfClasses):
        a[j]=perceptron(x[j],x[(j+1)%noOfClasses],a[j],eta)

    for i in range(len(y)):#noOfTestClasses
        for j in y[i]:#no of points in class i
            cl=[0 for k in range(noOfClasses)]
            for p in range(noOfClasses):
                val=np.asscalar(np.matmul(a[p],j))
                if val>=0:
                    cl[p]+=1
                else:
                    cl[(p+1)%noOfClasses]+=1
            maxv=cl[0]
            maxi=0
            for p in range(1,noOfClasses):
                if cl[p]>maxv:
                    maxv=cl[p]
                    maxi=p
            confM[i][maxi]+=1

    return confM


def plotPerc(a,x,m,n,eta):
    fig =  plt.figure()
    col=['orange','greenyellow','skyblue']
    col1=['red','green','blue']
    xx = np.arange(-20,25,1)
    yy = np.arange(-20,25,1)
    for i in xx:#noOfClasses
        for j in yy:#no of points in class i
            pt=[1,i,j]
            val=np.asscalar(np.matmul(a,pt))
            if val>=0:
                plt.plot(pt[1],pt[2],col[m],marker='.')
            else:
                plt.plot(pt[1],pt[2],col[n],marker='.')
    pt=[0 for i in range(2)]
    for i in x[m]:
        for p in range(2):
            pt[p]=i[p+1]
        plt.plot(pt[0],pt[1],col1[m],marker='.')
    for i in x[n]:
        for p in range(2):
            pt[p]=i[p+1]
        plt.plot(pt[0],pt[1],col1[n],marker='.')
    name='perc'+str(m)+'_'+str(n)+'_eta'+str(eta)+'.png'
    fig.savefig(name, dpi=fig.dpi)

def plotPerc2(x,eta):
    global noOfClasses
    fig = plt.figure()
    col=['orange','greenyellow','skyblue']
    col1=['red','green','blue']
    xx = np.arange(-20,25,0.3)
    yy = np.arange(-20,25,0.3)
    a=[[1 for p in range(dim+1)] for q in range(noOfClasses)]
    for j in range(noOfClasses):
        a[j]=perceptron(x[j],x[(j+1)%noOfClasses],a[j],eta)
    for p in xx:#noOfClasses
        for q in yy:#no of points in class i
            cl=[0 for i in range(noOfClasses)]
            pt=[1,p,q]
            for j in range(noOfClasses):
                val=np.asscalar(np.matmul(a[j],pt))
                if val>=0:
                    cl[j]+=1
                else:
                    cl[(j+1)%noOfClasses]+=1
            maxv=cl[0]
            maxi=0
            for j in range(1,noOfClasses):
                if cl[j]>maxv:
                    maxv=cl[j]
                    maxi=j
            plt.plot(p,q,col[maxi],marker='.')
    pt=[0 for i in range(2)]
    for i in range(noOfClasses):
        for j in x[i]:
            for p in range(2):
                pt[p]=j[p+1]
            plt.plot(pt[0],pt[1],col1[i],marker='.')
    name='percAll_eta'+str(eta)+'.png'
    fig.savefig(name, dpi=fig.dpi)


def main():
    global noOfClasses
    path='resultsPerc.txt'
    f=open(path,'w')
    global noOfEtaIter
    global eta
    global eta_diff
    x,y = inp()
    for i in range(noOfEtaIter):
        for j in range(noOfClasses):
            a=[1 for i in range(dim+1)]
            a=perceptron(x[j],x[(j+1)%noOfClasses],a,eta)
            plotPerc(a,x,j,(j+1)%noOfClasses,eta)
        eta+=eta_diff
main()
