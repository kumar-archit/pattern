import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import math
classes = 2
dim=2
clusters=30

def inp():
    pathTrain='train_75/'
    pathTest='test_25/?non_test.txt'

    pathMeans='?gmm_means*.txt'+str(clusters)
    pathCov='?gmm_cov*.txt'+str(clusters)
    pathPk='?gmm_pk*.txt'+str(clusters)
    pathclassesTrain='?non_train.txt'
    means=[]
    cov=[]
    pk=[]
    test=[]
    train=[]

    path=pathTrain+pathMeans
    files=glob.glob(path)
    j=0
    for file in sorted(files):
        f=open(file,'r')
        means.append([])
        for line in f:
            # print(line)
            c=[float(n) for n in line.split()]
            means[j].append(c)
        j+=1
        f.close()

    path=pathTrain+pathCov
    files=glob.glob(path)

    j=0
    for file in sorted(files):
        f=open(file,'r')
        cov.append([])
        for line in f:
            c=[float(n) for n in line.split()]
            cov[j].append(c)
        j+=1
        f.close()

    path=pathTrain+pathPk
    files=glob.glob(path)

    j=0
    for file in sorted(files):
        f=open(file,'r')
        pk.append([])
        for line in f:
            c=[float(n) for n in line.split()]
            pk[j].append(c[0])
        j+=1
        f.close()
    j=0
    files=glob.glob(pathTest)
    for file in sorted(files):
        test.append([])
        f=open(file,'r')
        for line in f:
            c=[float(n) for n in line.split()]
            test[j].append(c)
        j+=1

    path=pathTrain+pathclassesTrain
    files=glob.glob(path)
    j=0
    for file in sorted(files):
        train.append([])
        f=open(file,'r')
        for line in f:
            c=[float(n) for n in line.split()]
            train[j].append(c)
        j+=1

    return train,test,means,cov,pk

def N(X,mu,sig,dim):

    x = np.array(X)
    Sig=np.array(sig)
    Mu=np.array(mu)
    X_mean_diff=np.subtract(x,Mu)
    try:
        sig_inv = np.linalg.inv(Sig)
    except:
        print("Error in N: Cannot find inverse: singular matrix ")
        print(Sig)
        sys.exit()
        print('det_sig : ',np.linalg.det(Sig))
    det_sig = np.linalg.det(sig)
    if (det_sig<0):
        det_sig *=-1
    X_mean_diff_T=X_mean_diff[np.newaxis].T

    try:
        base=math.pow(2*(np.pi),dim/2.0)*math.sqrt(det_sig)
        if base==0:
            print("error : division by zero, base is zero in N() ")
            sys.exit()
    except:
        print("Problem in calculatin base : ")
        print("\ndet_sig",det_sig)
        print("\n")
        sys.exit()
    try :
        top1=np.matmul(X_mean_diff,(np.matmul(sig_inv,X_mean_diff_T)).tolist())
    except:
        print("Problem in getting top1\n")
        sys.exit()

    try :
        top=math.exp(-0.5*np.asscalar(top1))
    except:
        print("Problem in calculating exponent of top 1, top1 : "+str(top1))
        print("X_mean_diff : ")
        print(X_mean_diff)
        print("\ndet_sig")
        print(sig_inv)
        sys.exit()
    N_=1.0*top/base

    return N_

def gGMM(X,avg1,cov1,pk1,avg2,cov2,pk2,dim):
    p1=probOfGMMclass(avg1,cov1,pk1,X,dim)
    p2=probOfGMMclass(avg2,cov2,pk2,X,dim)
    return p1-p2

def probOfGMMclass(mean,cov,pk,j,dim):
    prob=0.0
    global clusters
    global classes

    for i in range(clusters):
        covDdim=[[0 for p in range(dim)]for q in range(dim)]
        for p in range(dim):
            for q in range(dim):
                covDdim[p][q]=cov[i][p*dim+q]

        prob+=pk[i]*N(j,mean[i],covDdim,dim)
    return prob
def getConfMat(test, means,cov,pk):
    global classes
    global dim
    cfmt=[[0 for i in range(classes)]for j in range(classes)]
    for i in range(len(test)):#for each classData
        for j in test[i]:#each obs in classData
            actualClass=i
            pro=[0 for q in range(classes)]
            for q in range(classes):
                pro[q]=probOfGMMclass(means[q],cov[q],pk[q],j,dim)
            maxv=pro[0]
            maxi=0
            for q in range(1,classes):
                if pro[q]>maxv:
                    maxv=pro[q]
                    maxi=q
            cfmt[actualClass][maxi]+=1
    return cfmt
def plotRegion(train,means,cov,pk):
    global cluster
    global dim
    global classes
    X=[]
    Y=[]
    col = ["ro","bo","go"]
    for i in range(classes):
        X.append([])
        Y.append([])
        for j in train[i]:
            X[i].append(j[0])
            Y[i].append(j[1])
    delta=0.1
    xx = np.arange(-15,15,delta)
    yy = np.arange(-15,15,delta)
    Xx, Yy = np.meshgrid(xx, yy)
    xxx1=[]
    xxx2=[]
    yyy1=[]
    yyy2=[]

    fig, ax = plt.subplots()
    fig, bx = plt.subplots()
    newCol=["yellow", "skyblue", "lawngreen"]
    for p in xx:
        for q in yy:
            gg=gGMM([p,q],means[0],cov[0],pk[0],means[1],cov[1],pk[1],dim)
            if(gg>0):
                xxx1.append(p)
                yyy1.append(q)
            else:
                xxx2.append(p)
                yyy2.append(q)
    ax.scatter(xxx1,yyy1,c=newCol[0],zorder=1)
    ax.scatter(xxx2,yyy2,c=newCol[1],zorder=1)
    ax.plot(X[0],Y[0],col[0],zorder=1)
    ax.plot(X[1],Y[1],col[1],zorder=1)
    bx.scatter(xxx1,yyy1,c=newCol[0],zorder=1)
    bx.scatter(xxx2,yyy2,c=newCol[1],zorder=1)
    bx.plot(X[0],Y[0],col[0],zorder=1)
    bx.plot(X[1],Y[1],col[1],zorder=1)
###############################ploting of contours############################################
    covDdim=[[0 for p in range(dim)]for q in range(dim)]

    for i in range(classes):
        for cclus in range(clusters):
            Z=[[0 for i in range(len(xx))] for j in range(len(yy))]
            for p in range(dim):
                for q in range(dim):
                    covDdim[p][q]=cov[i][cclus][p*dim+q]
            for p in range(len(xx)):
                for q in range(len(yy)):
                    Z[q][p]=N([xx[p],yy[q]],means[i][cclus],covDdim,dim)
            bx.contour(Xx,Yy,Z,3,colors='k')
            plt.title('bx')
    levels = np.arange(-1.0,1.5,0.25)
    for i in range(classes):
        Z=[[0 for i in range(len(xx))] for j in range(len(yy))]
        for p in range(len(xx)):
            for q in range(len(yy)):
                    Z[q][p]=probOfGMMclass(means[i],cov[i],pk[i],[xx[p],yy[q]],dim)
        bx.contour(Xx,Yy,Z,10,colors='m')
        ax.contour(Xx,Yy,Z,10,colors='m')


    strName='decisionRegion_lns'+str(clusters)
    plt.show()
def p(means,cov,pk):
    print(means)
    print('**********************************')
    print(cov)
    print('**********************************')
    print(pk)
    print('**********************************')
def main():
    train, test, means,cov,pk=inp()
    confMat=getConfMat(test, means,cov,pk)
    print(confMat)
main()
