import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

def mydist(p,s):
    from fastdtw import fastdtw
    return fastdtw(p, s)[0]


loop = np.loadtxt("data_train_loop_0.csv")
other = np.loadtxt("data_train_negative_1.csv")

train = np.concatenate((loop[5:-5], other[5:-5]))
test = np.concatenate((loop[:5], loop[-5:], other[:5], other[-5:]))

labels_train = [0] * int(len(train)/2) + [1] * int(len(train)/2)
labels_test = [0] * int(len(test)/2) + [1] * int(len(test)/2)


n_neighbors = 20
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(train, labels_train)

proba = clf.predict_proba(test)
predicted = clf.predict(test)

print(predicted)
print(proba)

plt.figure("9")
plt.plot(range(len(test[9])), test[9])

plt.show()

a = 0