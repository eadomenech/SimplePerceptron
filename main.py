from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from perceptron import SimplePerceptron


X, y = make_classification(
    n_features=2, # The total number of features.
    n_classes=2, # The number of classes.
    n_samples=20, # The number of samples.
    n_redundant=0, # The number of redundant features.
    n_clusters_per_class=1 # The number of clusters per class.
)

p = SimplePerceptron(X, y)
w = p.train()

list_x = []
list_y = []
list_c = []
for item, out in enumerate(y):
    list_x.append(X[item][0])
    list_y.append(X[item][1])
    if y[item] == 1:
        list_c.append('#2ca02c')
    else:
        list_c.append('#7f7f7f')

if w is not None:
    m = (w[2] - w[0]) / w[1]
    plt.plot([-2, -1, 0, 1, 2], [(-w[0]-w[1]*i)/w[2] for i in [-2, -1, 0, 1, 2]], 'b')

plt.scatter(list_x, list_y, c=list_c)
plt.show()
