import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io

from models.PCA import PCA

# importing mnist dataset
splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])

# taking 100 samples from each class
df=df.groupby('label').sample(n=100,random_state=42)

# flattening each image
images = np.array([
    Image.open(io.BytesIO(x['bytes'])).resize((28,28))
    for x in df['image']
])
X = images.reshape(len(images),-1)

# plotting the first 10 images from each class
fig, axes = plt.subplots(10,10, figsize=(15,15))
for i in range(10):
    for j in range(10):
        axes[i,j].imshow(images[i*100+j], cmap='gray')
        axes[i,j].axis('off')
plt.tight_layout()
plt.title('original dataset')
plt.show()

# PCA on the dataset
pca=PCA(X)
pca = PCA(X)
pca.centering()
pca.covariance_matrix()
pca.eigen_decomposition()
pca.sort_e_vectors()

# PCA using different number of prinicipal components
for i in range(10,784,50):
    pca.plot_reduced_images(i)
