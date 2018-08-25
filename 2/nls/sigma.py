import numpy as np
import random
from sklearn.cluster import KMeans
import glob

k=30
consideration = 1
path = 'train_75/'+str(consideration)+'non_train.txt'
f=open(path,'r')
train=[]
for line in f:
    c=[float(w) for w in line.split()]
    train.append(c)
tr=np.array(train)
dim=2
kmeans = KMeans(k,init='random').fit(tr)
f=open('train_75/'+str(consideration)+'kmeans_train.txt'+str(k),'w')
for i in range(k):
    for j in range(dim):
            f.write(str(kmeans.cluster_centers_[i][j])+" ")
    f.write("\n")
f.close()

els=[[] for i in range(k)]
for i in range(len(train)):
    els[kmeans.labels_[i]].append(train[i])
g=open('train_75/'+str(consideration)+'cov_train.txt'+str(k),'w')
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
h=open('train_75/'+str(consideration)+'pk_train.txt'+str(k),'w')
for x in els:
    pk=1.0*len(x)/total
    h.write(str(pk))
    h.write("\n")
h.close()
