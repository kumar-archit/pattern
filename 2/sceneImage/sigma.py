import numpy as np
import random
from sklearn.cluster import KMeans
import glob
k=5
dim=24
consideration = 'arch'
cons =1
path = 'arc/2b/train/'+consideration+'/*.txt'
files=glob.glob(path)
train=[]
for file in files:
    f=open(file,'r')
    for line in f:
        c=[float(w) for w in line.split()]
        train.append(c)
tr=np.array(train)
kmeans = KMeans(k,init='random').fit(tr)
f=open('arc/2b/train/'+str(cons)+'kmeans_train.txt','w')
for i in range(k):
    for j in range(dim):
            f.write(str(kmeans.cluster_centers_[i][j])+" ")
    f.write("\n")
f.close()

els=[[] for i in range(k)]
for i in range(len(train)):
    els[kmeans.labels_[i]].append(train[i])
g=open('arc/2b/train/'+str(cons)+'cov_train.txt','w')
for i in range(k):
    cov=[[0 for j in range(dim)] for m in range(dim)]

    for j in range(len(els[i])):
        for q in range(dim):
            for p in range(dim):
                cov[q][p] += (els[i][j][q] - kmeans.cluster_centers_[i][q])*(els[i][j][p] - kmeans.cluster_centers_[i][p])

    for q in range(dim):
        for p in range(dim):
            cov[q][p] /= len(els[i])
            g.write(str(cov[q][p])+ " ")
    g.write("\n")
g.close()

total=0
for ok in els:
    total +=len(ok)
h=open('arc/2b/train/'+str(cons)+'pk_train.txt','w')
for x in els:
    pk=1.0*len(x)/total
    h.write(str(pk))
    h.write("\n")
h.close()
