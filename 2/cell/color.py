import numpy as np
import math
import sys
import matplotlib
import glob
from PIL import Image
import scipy
from scipy.misc import toimage
from scipy.misc import imshow
from matplotlib import pyplot as plt
clusters=3
dim=2

def inp():
    global clusters
    global dim
    mean=[]
    cov=[[[0 for i in range(dim)] for j in range(dim)] for k in range(clusters)]
    fm=open('3gmm_means_train.txt','r')
    fcov=open('3gmm_cov_train.txt','r')

    for line in fm:
        c=[float(w) for w in line.split()]
        mean.append(c)
    mm=np.array(mean)
    i=0
    for line in fcov:
        c=[float(w) for w in line.split()]
        for p in range(dim):
            for q in range(dim):
                cov[i][p][q]=c[p*dim+q]
        i+=1
    return mean,cov

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

def gGMMN(X,avg1,cov1,avg2,cov2,dim):
    n1=N(X,avg1,cov1,dim)
    n2=N(X,avg2,cov2,dim)
    return n1-n2

def color(mean,cov):
    global clusters
    global dim
    color=[[84,84,84],[147,147,147],[168,168,168]]
    path='Test/*.txt'
    files=glob.glob(path)

    for file in files:
        w=0#go till 512-7+1=506
        h=0
        l=0
        all_pixel = [[[0,0,0] for i in range(512)]for j in range(512)]
        fd=open(file,'r')
        for line in fd:
            c=[float(w) for w in line.split()]
            maxv=N(c,mean[0],cov[0],dim)
            ind=0
            for j in range(1,clusters):
                    x=N(c,mean[j],cov[j],dim)
                    if x>maxv:
                        maxv=x
                        ind=j

            # color the [w][h] patch as color[ind]
            for m in range(7):
                for n in range(7):
                    all_pixel[h+n][w+m] = color[ind]
            l += 1
            h = int(l/506)
            w = l%506
        all=np.array(all_pixel)
        file2 = file.split('.')[0]
        file2+="_gmm.jpg"

        scipy.misc.imsave(file2, all)
def main():
    mean,cov=inp()
    color(mean,cov)
main()
