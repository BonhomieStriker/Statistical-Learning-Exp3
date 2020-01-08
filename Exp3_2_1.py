import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
# #######################################################################
#Task-a Data generation
cluster1=np.random.uniform(0.5,1.5,(200,200))
cluster2=np.random.uniform(3.5,4.5,(200,200))
cluster3=np.random.uniform(6.5,7.5,(200,200))
X=np.concatenate((cluster1,cluster2,cluster3),1).T
plt.figure(1)
plt.title('Original Data')
plt.grid(True)
plt.plot(X[:, 0], X[:, 1], 'k.')
plt.show()
# #######################################################################
#Task-b mean
mean0 = np.mean(X,0)
print('Mean value of data:',mean0)
print('________________')
# #######################################################################
#Task-c covariance
cov0 = np.cov(X,rowvar=False)
print('The Shape of Covariance Matrix:')
print(np.shape(cov0))
print('________________')
print('Covariance Matrix:')
print(cov0)
print('________________')
# #######################################################################
#Task-d eigs
eigvals,eigvecs = la.eig(cov0)
print('The Shape of Eigenvalues:')
print(np.shape(eigvals))
print("Eigenvalues:")
print(eigvals)
print('________________')
print('The Shape of Eigenvectors:')
print(np.shape(eigvecs))
print("Eigenvectors:")
print(eigvecs)
print('________________')
# #######################################################################
#Task-e sort
ratio = 0.99
order = np.argsort(-eigvals)
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

eig_sum = np.cumsum(eigvals)
eig_total = np.sum(eigvals)
p95 = np.where(eig_sum >= ratio*eig_total)
p95 = p95[0][0]
print('Position of 95% Eigenvalues:',p95)
print('________________')
# #######################################################################
#Task-f PCA
eigvecs99 = eigvecs[:, :p95]
X0 = X - mean0
new_data = np.dot(X0, eigvecs99)
print('The shape of new_data:',np.shape(new_data))
print('________________')

# #######################################################################
#Task-g K-means
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
ESTIMATORS = {
    "N_cluster2": KMeans(n_clusters=2),
    "N_cluster3": KMeans(n_clusters=3),
    "N_cluster4": KMeans(n_clusters=4)
}

k_means_cluster_centers = dict()
k_means_labels = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(new_data)
    # We want to have the same colors for the same cluster from the KMeans algorithm. Let's pair the cluster centers per
    # closest one.
    k_means_cluster_centers[name] = np.sort(estimator.cluster_centers_, axis=0)
    k_means_labels[name] = pairwise_distances_argmin(new_data, k_means_cluster_centers[name])

# plot
i = 1
fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['red', 'yellow', 'blue', 'black']
for name, estimator in ESTIMATORS.items():
    ax = fig.add_subplot(1, 3, i)
    for j in range(i + 1):
        ind = k_means_labels[name] == j
        cluster_center = k_means_cluster_centers[name][j, :]
        ax.plot(new_data[ind, 0], new_data[ind, 1], 'w',
                markerfacecolor=colors[j], marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=colors[j],
                markeredgecolor='k', markersize=6)
    ax.set_title(name)
    ax.set_xticks(())
    ax.set_yticks(())

    i = i + 1

plt.show()