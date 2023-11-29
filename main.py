import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import histogram, vector, downsample
from sklearn.neighbors import KNeighborsClassifier

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(labels), 28, 28).astype(np.float64)
    
    return images, labels

X_train, y_train = load_mnist('Data/', kind='train')
print('Train images shape:', X_train.shape)
print('Train labels shape:', y_train.shape)

X_t10k, y_t10k = load_mnist('Data/', kind='t10k')
print('Test images shape: ', X_t10k.shape)
print('Test labels shape: ', y_t10k.shape)
print()

print('Extracting vector...')
X_train_vector = vector.get_vector(X_train)
X_test_vector = vector.get_vector(X_t10k)
print('Train vector shape:', X_train_vector.shape)
print('Test vector shape:', X_test_vector.shape)
print()

print('Extracting downsample...')
X_train_downsample = downsample.get_downsample(X_train)
X_test_downsample = downsample.get_downsample(X_t10k)
print('Train downsample shape:', X_train_downsample.shape)
print('Test downsample shape:', X_test_downsample.shape)
print()

print('Extracting histogram...')
X_train_histogram = histogram.get_histogram(X_train)
X_test_histogram = histogram.get_histogram(X_t10k)
print('Train histogram shape:', X_train_histogram.shape)
print('Test histogram shape:', X_test_histogram.shape)
print()

dict = {}
for k in range(10):
    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(X_train_histogram, y_train)
    predictions = kNN.predict(X_test_histogram)
    accuracy = np.mean(predictions == y_t10k)
    dict[k] = accuracy
print(f"Accuracy: {dict}")

# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0]
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()
