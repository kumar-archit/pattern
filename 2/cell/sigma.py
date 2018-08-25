import numpy as np
import random
from sklearn.cluster import KMeans
import glob

path = '/home/archit/sem5/pr/ass2/arc/2c/all/*.txt'
files=glob.glob(path)
train=[]
for file in files:
    f=open(file,'r')
    for line in f:
        c=[float(w) for w in line.split()]
        train.append(c)
tr=np.array(train)
k=3
dim=2
kmeans = KMeans(k,init='random').fit(tr)
f=open('2c/3kmeans_train.txt','w')
for i in range(k):
    for j in range(dim):
            f.write(str(kmeans.cluster_centers_[i][j])+" ")
            # print(str(kmeans.cluster_centers_[i][j]))
    f.write("\n")
f.close()

els=[[] for i in range(k)]
# print(els)
for i in range(len(train)):
    els[kmeans.labels_[i]].append(train[i])
    # print(kmeans.labels_[i])
g=open('2c/3cov_train.txt','w')
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
h=open('2c/3pk_train.txt','w')
for x in els:
    pk=1.0*len(x)/total
    h.write(str(pk))
    h.write("\n")
h.close()
