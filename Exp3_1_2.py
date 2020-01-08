from time import time
import logging
import matplotlib.pyplot as plt
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy array

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# cov_face = np.cov(X_train,rowvar=False)
# print(np.shape(cov_face))
#
# eigval,eigvec = la.eig(cov_face)
# eigvec = eigvec.real
# eig_pairs = [(np.abs(eigval[i]), eigvec[:,i]) for i in range(len(eigval))]
# eig_pairs.sort(reverse=True)
# print(eig_pairs)
# # list = np.argsort(-eigval)
# # list = list[0:round(n_features*0.8)]
# # new_feature = eigvec[list]
# # print(np.shape(new_feature))
# # print(np.shape(X_train))
# # X_train = np.dot(new_feature,np.transpose(X_train))
# # print(np.shape(X_train))

#############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 500

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
# print('____________________________________')
# print('Explained Variance Ratio:')
# print(pca.explained_variance_ratio_)
# print('____________________________________')
var_sum = np.cumsum(pca.explained_variance_ratio_)
# print('Explained Variance Ratio:',var_sum)
var80 = np.where(var_sum >= 0.8)
p80 = var80[0][0]
var98 = np.where(var_sum >= 0.98)
p98 = var98[0][0]
print('Position of Sum Variance 80%:',p80)
print('Position of Sum Variance 98%:',p98)
print("done in %0.3fs" % (time() - t0))
print('____________________________________')
pca80 = PCA(n_components=p80, svd_solver='randomized',
          whiten=True).fit(X_train)
pca98 = PCA(n_components=p98, svd_solver='randomized',
          whiten=True).fit(X_train)
eigenfaces80 = pca80.components_.reshape((p80, h, w))
eigenfaces98 = pca98.components_.reshape((p98, h, w))
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca80 = pca80.transform(X_train)
X_test_pca80 = pca80.transform(X_test)
X_train_pca98 = pca98.transform(X_train)
X_test_pca98 = pca98.transform(X_test)
print("done in %0.3fs" % (time() - t0))
print('The shape of eigenfaces with 80% variance:',np.shape(eigenfaces80))
print('The shape of eigenfaces with 98% variance:',np.shape(eigenfaces98))
# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
print('____________________________________')
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf80 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf80 = clf80.fit(X_train_pca80, y_train)

print("Best estimator found by grid search(80%):")
print(clf80.best_estimator_)
print('____________________________________')
clf98 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf98 = clf98.fit(X_train_pca98, y_train)

print("Best estimator found by grid search(98%):")
print(clf98.best_estimator_)
print('____________________________________')
# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
y_pred80 = clf80.predict(X_test_pca80)
y_pred98 = clf98.predict(X_test_pca98)

print('80%')
print(classification_report(y_test, y_pred80, target_names=target_names))
print(confusion_matrix(y_test, y_pred80, labels=range(n_classes)))
print('____________________________________')
print('98%')
print(classification_report(y_test, y_pred98, target_names=target_names))
print(confusion_matrix(y_test, y_pred98, labels=range(n_classes)))
print('____________________________________')
#############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, start = 0, n_row=3, n_col=4  ):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i+start].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles80 = [title(y_pred80, y_test, target_names, i)
                     for i in range(y_pred80.shape[0])]

plot_gallery(X_test, prediction_titles80, h, w)

# plot the gallery of the most significative eigenfaces
eigenface_titles80 = ["eigenface %d" % (i+1) for i in range(eigenfaces80.shape[0])]
plot_gallery(eigenfaces80, eigenface_titles80, h, w)
plt.show()



prediction_titles98 = [title(y_pred98, y_test, target_names, i)
                     for i in range(y_pred98.shape[0])]

plot_gallery(X_test, prediction_titles98, h, w)

# plot the gallery of the most significative eigenfaces
eigenface_titles98 = ["eigenface %d" % (i+p80+1) for i in range(eigenfaces98.shape[0])]
plot_gallery(eigenfaces98, eigenface_titles98, h, w,p80+1)
plt.show()
