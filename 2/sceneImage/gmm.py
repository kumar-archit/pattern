import numpy as np
import math
import glob
import sys
k=4
cons=1
consideration='arch'
path = 'train/'+consideration+'/*.txt'
files=glob.glob(path)

def N(X,mu,sig,k):
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
        base=math.pow(2*(np.pi),k/2.0)*math.sqrt(det_sig)
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

def resp(mu,sig,pi,k,dim,X):
    gamma=[[0 for i in range(k)]for j in range(len(X))]
    sum=0.0
    for i in range(len(X)):
        s=0.0
        for j in range(k):
            s+=pi[j]*N(X[i],mu[j],sig[j],dim)
        for j in range(k):
            gamma[i][j] = (pi[j]/s)*N(X[i],mu[j],sig[j],dim)
    return gamma


def M(mu,sig,pi,k,dim,X,gamma):

    for i in range(k):
        s=0.0
        for j in range(len(X)):
            s+=gamma[j][i]

        for l in range(dim):
            mu[i][l] = 0.0
        for m in range(dim):
            for l in range(dim):
                sig[i][m][l] =0.0
        pi[i]=s/len(X)


        for j in range(len(X)):
            X_new=[0 for bb in range(dim)]
            for bb in range(dim):
                X_new[bb]=X[j][bb]*gamma[j][i]
            for p in range(dim):
                mu[i][p]=mu[i][p]+X_new[p]/s

        for j in range(len(X)):
            X_mean_diff = (np.subtract(X[j],mu[i])).tolist()
            sig_ini = (np.outer(X_mean_diff, X_mean_diff)).tolist()

            for p in range(len(sig[i])):
                for q in range(len(sig[i])):
                    sig_ini[p][q] = sig_ini[p][q]*gamma[j][i]
                    sig[i][p][q] = sig[i][p][q]+sig_ini[p][q]
        for dim1 in range(len(sig[i])):
            for dim2 in range(len(sig[i])):
                sig[i][dim1][dim2] = 1.0*sig[i][dim1][dim2]/s

def l(X,mu,sig,pi,k,dim):
    ll=0.0
    for i in range(len(X)):
        t=0.0
        for j in range(k):
            tt = (N(X[i],mu[j],sig[j],dim))
            if tt>1:
                print("In func l : probability is greater than 1, P :"+str(tt))
                print("N : "+str(N(X[i],mu[j],sig[j],dim)))
                sys.exit()
            tt*=pi[j]
            t+=tt
        try:
            if t>1:
                print("In func l : probability is greater than 1 ,P : "+str(t))
                sys.exit()
            ll+=math.log(t)
        except:
            print("In function l: problem in getting log, arguement out of range")
            print("argument for log : "+str(t))
            sys.exit()
    return ll

def GMM(mu,sig,pi,X,k,dim):
    for i in range(k):
        if np.linalg.det(sig[i])==0:
            print("singular mat found at pos : ", i)
            print("\n")
            diag_avg =0.0
            for p in range(dim):
                diag_avg += sig[i][p][p]
            diag_avg /= dim
            for j in range(dim):
                for ll in range(dim):
                    if j==ll and sig[i][j][j] == 0:
                        sig[i][j][j]+=diag_avg
            if np.linalg.det(sig[i])==0:
                for j in range(dim):
                    for ll in range(dim):
                        if j!=ll:
                            sig[i][j][ll]=0


    l_theta_old=l(X,mu,sig,pi,k,dim)
    print("iteration : ", i)
    print("log likelihood : ", l_theta_old)


    for z in range(20):
        gamma =resp(mu,sig,pi,k,dim,X)
        M(mu,sig,pi,k,dim,X,gamma)


        for i in range(k):
            if np.linalg.det(sig[i])==0:
                print("singular mat found at pos : ", i)
                print("\n")
                diag_avg =0.0
                for p in range(dim):
                    diag_avg += sig[i][p][p]
                diag_avg /= dim
                for j in range(dim):
                    for ll in range(dim):
                        if j==ll and sig[i][j][j] == 0:
                            sig[i][j][j]+=diag_avg
                if np.linalg.det(sig[i])==0:
                    for j in range(dim):
                        for ll in range(dim):
                            if j!=ll:
                                sig[i][j][ll]=0

        arc = open('train/'+str(cons)+'gmm_means_train.txt', 'w')
        for i in range(len(mu)):
            for j in range(len(mu[0])):
                arc.write(str(mu[i][j]))
                arc.write(' ')
            arc.write("\n")
        arc.close()

        arc2 = open('train/'+str(cons)+'gmm_cov_train.txt', 'w')
        for i in range(len(sig)):
            for j in range(dim):
                for q in range(dim):
                    arc2.write(str(sig[i][j][q])+" ")
            arc2.write("\n")
        arc2.close()

        arc3 = open('train/'+str(cons)+'gmm_pk_train.txt', 'w')
        for i in range(len(pi)):
            arc3.write(str(pi[i])+" ")
            arc3.write("\n")
        arc3.close()

        l_theta_new=l(X,mu,sig,pi,k,dim)
        print("updated log likelihood : ", l_theta_new)
        if abs(l_theta_old-l_theta_new)<0.1:
            print("no of iterations : "+str(z))
            break
        l_theta_old=l_theta_new

def main():
    global k
    global cons
    global consideration
    dim=24
    X=[]
    for file in files:

        e=open(file,'r')
        for line in e:
            c=[float(w) for w in line.split()]
            X.append(c)


    mu,sig,pk=[],[],[]
    f=open('train/'+str(cons)+'kmeans_train.txt','r')
    g=open('train/'+str(cons)+'cov_train.txt','r')
    h=open('train/'+str(cons)+'pk_train.txt','r')
    for line in f:
        mu.append([float(n) for n in line.split()])

    for line in g:
        c=[float(n) for n in line.split()]
        d=[[0 for i in range(dim)] for i in range(dim)]
        for i in range(dim):
            for j in range(dim):
                d[i][j]=c[i*dim+j]
        sig.append(d)

    pi=[]
    for line in h:
        pi.append(float(line))

    GMM(mu,sig,pi,X,k,dim)
    arc = open('train/'+str(cons)+'gmm_means_train.txt', 'w')
    for i in range(len(mu)):
        for j in range(len(mu[0])):
            arc.write(str(mu[i][j]))
            arc.write(' ')
        arc.write("\n")
    arc.close()

    arc2 = open('train/'+str(cons)+'gmm_cov_train.txt', 'w')
    for i in range(len(sig)):
        for j in range(dim):
            for q in range(dim):
                arc2.write(str(sig[i][j][q])+" ")
        arc2.write("\n")
    arc2.close()

    arc3 = open('train/'+str(cons)+'gmm_pk_train.txt', 'w')
    for i in range(len(pi)):
        arc3.write(str(pi[i])+" ")
        arc3.write("\n")
    arc3.close()


main()
