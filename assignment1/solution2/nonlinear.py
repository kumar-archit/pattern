import glob
import numpy as np
import matplotlib.pyplot as plt

path = "/home/koti/PycharmProjects/pattern/assignment1/solution2/class175.txt"
x = []
files = glob.glob(path)
i = 0
j = 0
for file in files:
    x.append([])
    f = open(file, 'r')
    for line in f:
        c, d = line.split()
        x[i].append([])
        x[i].append([])
        x[i][0].append(float(c))
        x[i][1].append(float(d))
        j = j + 1
    f.close()
    i += 1
i = 0
s1x = 0
s1y = 0
while i < j:
    s1x = s1x + x[0][0][i]
    s1y = s1y + x[0][1][i]
    i += 1
plt.figure(1)
i = 0
while i < 1:
    plt.scatter(x[i][0], x[i][1])
    i += 1



print(s1x/j)

print(s1y/j)

plt.figure(2)
path = "/home/koti/PycharmProjects/pattern/assignment1/solution2/class275.txt"

x = []
files = glob.glob(path)
i = 0
j = 0
for file in files:
    x.append([])
    f = open(file, 'r')
    for line in f:
        c, d = line.split()
        x[i].append([])
        x[i].append([])
        x[i][0].append(float(c))
        x[i][1].append(float(d))
        j = j + 1
    f.close()
    i += 1
i = 0
s2x = 0
s2y = 0
while i < j:
    s2x = s2x + x[0][0][i]
    s2y = s2y + x[0][1][i]
    i += 1


print(s2x/j)
print(s2y/j)

i = 0
while i < 1:
    plt.scatter(x[i][0], x[i][1])
    i += 1

ax = plt.gca()
ax.set_xticklabels([])

plt.show()


