import numpy as np
import pandas as pd
import warnings
from collections import Counter
import random
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')


def k_nearest_neig( data, predict, k):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')

    distances = []

    for group in data:
        for features in data[group]:
            euclidean_dis = np.linalg.norm(np.array(features) - np.array(predict))
            #linalg mean linear algebra library and norm use for all othe sqrt and other calcuations
            distances.append([euclidean_dis, group])

    vote = [ j[1] for j in sorted(distances)[:k]]
    #print(Counter(vote).most_common(1)[0])
    vote_result = Counter(vote).most_common(1)[0][0]

    return vote_result   #it will return the group of the newly classified data



#Main program
data = pd.read_csv('iris.data.txt')
dataset = data.values.tolist()
random.shuffle(dataset)

train_set = { 'Iris-setosa' : [], 'Iris-versicolor': [], 'Iris-virginica':[] }
test_set = { 'Iris-setosa' : [], 'Iris-versicolor': [], 'Iris-virginica':[] }

test_size = 0.5
train_data = dataset[:int( test_size * len(dataset) )]
test_data = dataset[int( test_size * len(dataset) ):]

for data in train_data:
    train_set[data[-1]].append(data[:-1])

for data in test_data:
    test_set[data[-1]].append(data[:-1])


correct = 0
total = 0
c = tuple(np.random.rand(4))

flower_set = { 'Iris-setosa' : '#9e1010', 'Iris-versicolor': '#10319e',
                'Iris-virginica': '#369e10' }

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neig( train_set ,data, k=7)
        if vote == group:
            correct +=1
        ax.scatter(data[0], data[1], data[2],s=30, color=flower_set[vote])
        total +=1


ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')

plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], loc=2)
az = plt.gca()
leg = az.get_legend()
leg.legendHandles[0].set_color('#9e1010')
leg.legendHandles[1].set_color('#10319e')
leg.legendHandles[2].set_color('#369e10')

acc = (correct / total) * 100

print('Accuracy:', acc, '%')
plt.show()
