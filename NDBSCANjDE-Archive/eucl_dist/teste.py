import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from cpu_dist import dist
from sklearn.metrics.pairwise import euclidean_distances
#from gpu_dist import dist as gdist

a = np.random.rand(10000,3)
b = np.random.rand(10000,3)
#print(np.allclose(pairwise_distances(a,b, 'sqeuclidean'), dist(a,b), atol=1e-5))
#print(a, b)

c = np.asarray([1, 2])
d = np.asarray([3, 4])
print(c, d)
#print(dist(c,d))
#print(np.allclose(pairwise_distances(a,b, 'sqeuclidean'), gdist(a,b), atol=1e-5))

#a = np.random.rand(800,2048).astype(np.float32)
#b = np.random.rand(800,2048).astype(np.float32)
#print(np.allclose(pairwise_distances(a,b, 'sqeuclidean'), dist(a,b), atol=1e-5))


X = [[1.5214, 1.5214]]
# distance between rows of X
print(euclidean_distances(X, X))


# get distance to origin
print(euclidean_distances(X, [[1.9350, 1.9350]]))
