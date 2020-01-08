import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
# #######################################################################
#Task-a Data
cluster1=np.random.uniform(0.5,1.5,(2,200))
cluster2=np.random.uniform(3.5,4.5,(2,200))
cluster31=np.random.uniform(3.5,4.5,(1,200))
cluster32=np.random.uniform(0.5,1.5,(1,200))
cluster3=np.concatenate((cluster31,cluster32))
X=np.concatenate((cluster1,cluster2,cluster3),1).T
# #######################################################################
#Task-b mean
mean0 = np.mean(X,0)
mean1 = np.mean(cluster1,1)
mean2 = np.mean(cluster2,1)
mean3 = np.mean(cluster3,1)
X0 = X - mean0
# #######################################################################
#Task-c covariance
cov0 = np.cov(X,rowvar=False)
print('Covariance Matrix:')
print(cov0)
print('————————')
# #######################################################################
#Task-d eigs
w,v = la.eig(cov0)
print("Eigenvalues:")
print(w)
print('————————')
print("Eigenvectors:")
print(v)
print('————————')
# #######################################################################
#Task-e sort
eig_pairs = [(np.abs(w[i]), v[:,i]) for i in range(len(w))]
eig_pairs.sort(reverse=True)
new_feature=eig_pairs[0][1]
# print(np.shape(eig_pairs))
new_data = np.dot(new_feature,np.transpose(X0))
# print(np.shape(X0))
plt.figure(1)
plt.plot(X[:,0],X[:,1],'r.')
plt.plot(new_data,[3]*len(new_data),'b.')
plt.legend(["Original Data","Dim-reduced data"])
plt.grid(True)
plt.show()
