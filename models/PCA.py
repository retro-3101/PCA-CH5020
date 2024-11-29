import numpy as np
import matplotlib.pyplot as plt

class PCA:

    def __init__(self, X):
        self.X = X
        self.n = len(X)        
        self.d = len(X[0])      

    def centering(self):
        self.X_centered = self.X - np.mean(self.X, axis=0)  
    
    def covariance_matrix(self):
        self.C_X = np.cov(self.X_centered, rowvar=False)  
    
    def eigen_decomposition(self):
        self.e_values, self.e_vectors = np.linalg.eig(self.C_X)

    def sort_e_vectors(self):
        sorted_indices = np.argsort(self.e_values)[::-1]
        self.e_values = self.e_values[sorted_indices]
        self.e_vectors = self.e_vectors[:, sorted_indices]

    def transform(self, k):
        self.W = self.e_vectors[:, :k]
        X_reduced = self.X_centered @ self.W
        return X_reduced
    
    def variance_explained(self, k):
        total_var = np.sum(self.e_values)
        explained_var = np.sum(self.e_values[:k])
        ratio = explained_var/total_var
        return ratio
    
    def reduce_images(self,k):
        X_reduced = self.transform(k)
        
        ratio = np.real(self.variance_explained(k))
        np.set_printoptions(precision=4, suppress=True)
        print(f'varaince explained by {k} principal components is {ratio*100}%')

        reduced_img = X_reduced @ self.W.T
        return reduced_img
    
    def plot_reduced_images(self,k):
        reduced_imgs = np.real(self.reduce_images(k))
        reduced_imgs = reduced_imgs.reshape(-1,28,28)
        fig, axes = plt.subplots(10,10, figsize=(15,15))
        for i in range(10):
            for j in range(10):
                axes[i,j].imshow(reduced_imgs[i*100+j], cmap='gray')
                axes[i,j].axis('off')
        plt.tight_layout()
        plt.savefig(f"./pca_images/PCA_{k}.png", bbox_inches='tight')
        plt.show()