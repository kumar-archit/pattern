import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import math
classes = 3
dim=24
clusters=[4,4,4]
def inp():
    pathTrain='train/'
    pathTest='test/'
    folders=["arch/*.txt","forest_path/*.txt","highway/*.txt"]
    pathMeans='?gmm_means*.txt'
    pathCov='?gmm_cov*.txt'
    pathPk='?gmm_pk*.txt'

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
            c=[float(n) for n in line.split()]
            means[j].append(c)
        j+=1
        f.close()

    path=pathTrain+pathCov
    files=glob.glob(path)
    i=0
    for file in sorted(files):
        f=open(file,'r')
        cov.append([])#for each class
        for line in f:
            c=[float(n) for n in line.split()]
            cov[i].append(c)
        i+=1
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

    for i in range(classes):
        path=pathTest+folders[i]
        test.append([])
        files=glob.glob(path)
        j=0
        for file in sorted(files):
            test[i].append([])
            f=open(file,'r')
            for line in f:
                c=[float(n) for n in line.split()]
                test[i][j].append(c)
            j+=1

    for i in range(classes):
        path=pathTrain+folders[i]
        files=glob.glob(path)
        j=0
        train.append([])
        for file in sorted(files):
            train[i].append([])
            f=open(file,'r')
            for line in f:
                c=[float(n) for n in line.split()]
                train[i][j].append(c)
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
        print("X_mean_diff",X_mean_diff)
        print("sig_inv",sig_inv)
        # if -0.5*
        sys.exit()

    try :
        top=math.exp(-0.5*np.asscalar(top1))
    except:
        if -0.5*np.asscalar(top1)>7.0937e2:
            top=1.1898072959500533e+308
    N_=1.0*top/base

    return N_

def g_i(X,avg ,cov, dim,pk):#returns g_i(x=X)#parameters: d-dim X,avg,k==d,pk
    n=N(X,avg,cov,dim)
    n=math.log(n)+math.log(pk)
    return n
def gGMM(X,avg1,cov1,pk1,avg2,cov2,pk2,dim):
    p1=probOfGMMclass(avg1,cov1,pk1,X,dim)
    p2=probOfGMMclass(avg2,cov2,pk2,X,dim)
    return p1-p2

def probOfGMMclass(mean,cov,pk,clusters,j,dim):
    prob=0.0

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
    global clusters
    cfmt=[[0 for i in range(classes)]for j in range(classes)]
    for i in range(len(test)):#for each class
        for j in range(len(test[i])):#each obs/scene image in class
            actualClass=i
            proImage=[0 for q in range(len(test[i][j]))]
            for k in range(len(test[i][j])):
                pro=[0 for q in range(classes)]
                for q in range(classes):
                    pro[q]=probOfGMMclass(means[q],cov[q],pk[q],clusters[q],test[i][j][k],dim)
                maxv=pro[0]
                maxi=0
                for q in range(1,classes):
                    if pro[q]>maxv:
                        maxv=pro[q]
                        maxi=q
                proImage[maxi]+=1
            maxv=proImage[0]
            maxi=0
            for q in range(1,len(test[i][j])):
                if proImage[q]>maxv:
                    maxv=pro[q]
                    maxi=q
            cfmt[actualClass][maxi]+=1
    return cfmt

def main():
    train, test, means,cov,pk=inp()
    confMat=getConfMat(test, means,cov,pk)
    print(confMat)

main()
