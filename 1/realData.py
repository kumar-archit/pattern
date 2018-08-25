import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import glob
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import math
import sys


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

def g_i(X,avg ,cov, dim,pk):#returns g_i(x=X)#parameters: d-dim X,avg,k==d,pk
    n=N(X,avg,cov,dim)
    n=math.log(n)+math.log(pk)
    return n


def g(X,avg1,cov1,pk1,avg2,cov2,pk2,dim):
    col = ["olive","cyan","orange"]
    try:
        n1=math.log(N(X,avg1,cov1,dim)) + math.log(pk1)
    except:
        print("problem in func g: ", X)
        print("N(x) : ", N(X,avg1,cov1,dim))
        print("pk1 : ",pk1)
    try:
        n2=math.log(N(X,avg2,cov2,dim)) + math.log(pk2)
    except:
        print("problem in func g: ", X)
        print("N(x) : ", N(X,avg2,cov2,dim))
        print("pk2 : ",pk2)
    return (n1-n2)

def findCovMat(x,dim):
    mean=[0.0 for i in range(dim)]
    for i in range(len(x)):
        for j in range(dim):
            mean[j] += x[i][j]/len(x)
    cov=[[0 for p in range(dim)] for q in range(dim)]

    for i in range(len(x)):
        for p in range(dim):
            for q in range(dim):
                cov[p][q]+=(x[i][p]-mean[p])*(x[i][q]-mean[q])/len(x)
    return cov

def findMeanVec(x,dim):
    mean=[0.0 for i in range(dim)]
    for i in range(len(x)):
        for j in range(dim):
            mean[j] += x[i][j]/len(x)
    return mean
#k*k
def main():
    path = 'rd_group13/Train/*.txt'
    files=glob.glob(path)
    dim=2
    classes=3
    tot=0
    x=[]
    n=[0 for i in range(classes)]
    for i in range(classes):
        x.append([])
    i=0
    allCovFull=[]
    allCovDiag=[ [[0 for f in range(dim)] for g in range(dim)] for h in range(classes)]
    allMean=[]
    for file in sorted(files):
        f=open(file,'r')
        for line in f:
            c=[float(n) for n in line.split()]
            x[i].append(c)
            n[i] += 1
        tot+=n[i]

    # finding originally cov matrices of classes
        allCovFull.append(findCovMat(x[i], dim))
        allMean.append(findMeanVec(x[i],dim))
        f.close()
        i+=1

    sameCovDiag=[[0 for p in range(dim)] for q in range(dim)]
    sameCovFull=[[0 for p in range(dim)] for q in range(dim)]

    for i in range(classes):
        for p in range(dim):
            for q in range(dim):
                sameCovFull[p][q]+=allCovFull[i][p][q]/classes
                if p==q:
                    allCovDiag[i][p][q]=allCovFull[i][p][q]
    sumCov=0.0
    for p in range(dim):
        for q in range(dim):
            sumCov+=sameCovFull[p][q]
    sumCov /= dim*dim
    for p in range(dim):
        sameCovDiag[p][p]=sumCov
    X=[]
    Y=[]
    col = ["ro","bo","go"]
    for i in range(classes):
        X.append([])
        Y.append([])
        for j in range(len(x[i])):
            X[i].append(x[i][j][0])
            Y[i].append(x[i][j][1])

    xx = np.arange(-1,1000,5)
    yy = np.arange(400,2500,5)

    for i in range(classes):
        xxx1=[]
        xxx2=[]
        yyy1=[]
        yyy2=[]
        for p in xx:
            for q in yy:
                gg=g([p,q],allMean[i],allCovFull[i],n[i]/tot,allMean[(i+1)%classes],allCovFull[(i+1)%classes],n[(i+1)%classes]/tot,dim)
                if(gg>0):
                    xxx1.append(p)
                    yyy1.append(q)
                else:
                    xxx2.append(p)
                    yyy2.append(q)
        plt.scatter(xxx1,yyy1,c="olive")
        plt.scatter(xxx2,yyy2,c="cyan")
        plt.plot(X[i],Y[i],col[i])
        plt.plot(X[(i+1)%classes],Y[(i+1)%classes],col[(i+1)%classes])
        strName='decisionRegion_allCovFull'
        plt.savefig(strName+str(i))
        plt.show()

    xxx=[]
    yyy=[]
    for p in range(classes):
        xxx.append([])
        yyy.append([])

    for p in xx:
        for q in yy:
            maxv=-1
            maxi=0

            pt=[p,q]
            for dd in range(classes):
                gVal=math.exp(g_i(pt,allMean[dd] ,allCovFull[dd], dim,n[dd]/tot))
                if gVal>maxv:
                    maxv=gVal
                    maxi=dd
            xxx[maxi].append(p)
            yyy[maxi].append(q)

    plt.scatter(xxx[0],yyy[0],c="orange")
    plt.scatter(xxx[1],yyy[1],c="cyan")
    plt.scatter(xxx[2],yyy[2],c="olive")
    plt.plot(X[0],Y[0],col[0])
    plt.plot(X[1],Y[1],col[1])
    plt.plot(X[2],Y[2],col[2])

    strName='allClassdecisionRegionAllCovFull'
    plt.savefig(strName)
    plt.show()

    path = 'rd_group13/Test/*.txt'

    files=glob.glob(path)
    i=0
    test=[]
    for m in range(classes):
        test.append([])

    for file in sorted(files):
        f=open(file,'r')
        for line in f:
            c=[float(n) for n in line.split()]
            test[i].append(c)
        i+=1
        f.close()

    cnt=[[0 for i in range(classes)] for j in range(classes)]

    print("Confusion matrix:")
    to=0
    correct=0

    for i in range(classes):
        for j in test[i]:
            to+=1
            maxe=0
            maxi=-100000000000
            for p in range(classes):
                if g_i(j,allMean[p] ,allCovFull[i], dim,n[p]/tot)>maxi:
                    maxe=p
                    maxi=g_i(j,allMean[p] ,allCovFull[i], dim,n[p]/tot)
            cnt[i][maxe] += 1

        correct+=cnt[i][i]

        for p in range(classes):
            print(cnt[i][p],end=' ')
        print("")

    print("classification accuracy:",correct/to)

    pr=0
    rec=0
    fm=0

    for i in range(classes):
        to=0
        to1=0
        for j in range(classes):
            to+=cnt[j][i]
            to1+=cnt[i][j]
    for i in range(classes):
        print("Precision of class",i+1,"=",cnt[i][i]/to1)
        pr+=cnt[i][i]/to1

        print("Recall of class",i+1,"=",cnt[i][i]/to)
        rec+=cnt[i][i]/to

        print("F-Measure of class",i+1,"=",2*(cnt[i][i]/to1)*(cnt[i][i]/to)/((cnt[i][i]/to1)+(cnt[i][i]/to)))

        fm+=2*(cnt[i][i]/to1)*(cnt[i][i]/to)/((cnt[i][i]/to1)+(cnt[i][i]/to))
        print("")

    print("Mean precision=",pr/classes)
    print("Mean recall=",rec/classes)
    print("Mean F-Measure=",fm/classes)
    #
    # # #########################for contour plot ############################
    # #
    xx = np.arange(-1,1000,5)
    yy = np.arange(400,2500,5)
    Xtrain=[]
    Ytrain=[]


    for i in range(classes):
        col = ["ro","bo","go"]
        Xtrain.append([])
        Ytrain.append([])
        for j in range(len(x[i])):
            Xtrain[i].append(x[i][j][0])
            Ytrain[i].append(x[i][j][1])
        plt.plot(Xtrain[i],Ytrain[i],col[i],marker='.')
        Z=[[0 for i in range(len(xx))] for j in range(len(yy))]

        for j in range(len(xx)):
            for q in range(len(yy)):
                Z[q][j]=N([xx[j],yy[q]],allMean[i],allCovFull[i],dim)
        plt.contour(xx,yy,Z,10)
    plt.savefig('contour_allCovFull')
    plt.show()

main()
