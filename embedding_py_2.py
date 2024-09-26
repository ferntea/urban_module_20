import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
# from umap import UMAP
from umap.umap_ import UMAP
# import UMAP
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

# import keras
from keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import sys
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def histogram_plot():

    fig = plt.figure(figsize=(6, 4))  # Create a figure

    # Define custom bin edges and labels for the feature ranges
    # bin_edges = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]  # Custom bin edges
    bin_edges = list(map(lambda x: 0.5 + 0.5 * x, range(int((8 - 0.5) / 0.5) + 1)))

    # bin_labels = ['1', '2', '3', '4', '5', '6', '7', '8']  # Labels for the bins
    # bin_labels = [str(i) for i in range(1, 9)]

    # Create a histogram based on y
    plt.hist(y.flatten(), bins=bin_edges, color='blue')
        # could also be edgecolor='black', width=0.5
    plt.xlabel('Light fastness')
    plt.ylabel('Frequency')
    plt.title('Histogram of y (Light Fastness)')
    # plt.xticklabels(bin_labels)  # make a test in future, seems not similar to automatic labels
    plt.grid(False)

    # plt.tight_layout()
    return


def tsne(X_train, y_train, model):
    # t-SNE
    # Import necessary libraries
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.manifold import TSNE

    # Perform t-SNE transformation
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, learning_rate=50)

    # Extract the embeddings from the model
    embeddings = model.layers[0](X_train).numpy()

    X_embedded = tsne.fit_transform(embeddings)

    # Plot the t-SNE visualization
    plt.figure(figsize=(6, 4))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train, cmap='seismic', s=70, alpha=0.90, edgecolor='black', linewidths=0.5)
    plt.colorbar()
    plt.title('t-SNE Visualization of Embeddings', fontsize=12)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.show()

    return X_embedded


def umap(X_train, y_train, model):
    # UMap
    # import numpy as np
    # import matplotlib.pyplot as plt
    # ...
    import umap

    # Assuming X and y are your input features and target labels

    # Generate UMAP embeddings for X_train
    umap_obj = UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='correlation', spread=0.9, learning_rate=500,
                         init='spectral', random_state=42)  # Creating UMAP object

    X_train_umap = umap_obj.fit_transform(X_train)

    # Plot the UMAP embeddings
    plt.figure(figsize=(6, 4))
    plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap='seismic', s=70, alpha=0.90, edgecolor='black', linewidths=0.5)
    plt.colorbar()
    plt.title('UMAP Visualization of Data with Target Labels', fontsize=12)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.show()

    return X_train_umap

# =-=-=-=-=-=-=-=-=-
# Load the dataset

dataset = pd.read_csv('compounds.csv')
smiles = dataset['SMILES']
targets = dataset['target_variable']

# print(smiles)

# Convert SMILES strings to RDKit molecules
molecules = [Chem.MolFromSmiles(smi) for smi in smiles]

# Calculate molecular descriptors using RDKit
descriptors = [AllChem.GetMorganFingerprintAsBitVect(
    mol,
    8,
    useFeatures=True,
    nBits=4096
) for mol in molecules]

# Normalize descriptors the target properties
X = np.array(descriptors)
y = np.array(targets)
# y = np.log10(y)

##=-=-=-=-=-=-=-=-=-=-=
import matplotlib.pyplot as plt

# Create embeddings and corresponding figures

def plot_embedding(X_embedded, title, filename):
    plt.figure(figsize=(6, 4))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=70, alpha=0.90,
                edgecolor='black', linewidths=0.5)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.savefig(filename)

# =-=-=-=-=-=-=-=-=-=-=-=
# t-SNE embedding

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2,
            perplexity=10,
            n_iter=1000,
            n_iter_without_progress=150,
            n_jobs=2,
            random_state=0
)
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne, 't-SNE Embedding', 'fig_tsne.png')

# =-=-=-=-=-=-=-=-=-=-=
# UMAP embedding
umap_emb = UMAP(n_components=2,
                n_neighbors=30,
                min_dist=0.1,
                metric='euclidean',
                spread=0.99,
                # learning_rate=500,
                init='pca',
                random_state=0)
X_umap = umap_emb.fit_transform(X)
plot_embedding(X_umap, 'UMAP Embedding', 'fig_umap.png')

##=-=-=-=-=-=-=-=-=-=-=
# Spectral embedding

from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt

spectral = SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
X_spec = spectral.fit_transform(X)
plot_embedding(X_spec, 'Spectral Embedding', 'fig_spectral.png')

##=-=-=-=-=-=-=-=-=-=-=
# Random Trees embedding

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

pipeline = make_pipeline(
    RandomTreesEmbedding(n_estimators=200, max_depth=2, random_state=0),
    TruncatedSVD(n_components=2)
)
X_rt = pipeline.fit_transform(X)
plot_embedding(X_rt, 'Random Trees Embedding', 'fig_rt.png')

##=-=-=-=-=-=-=-=-=-=-=
# MDS embedding

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

mds = MDS(n_components=2, n_init=4, max_iter=300, n_jobs=2, random_state=0)
X_mds = mds.fit_transform(X)
plot_embedding(X_mds, 'RMDS Embedding', 'fig_mds.png')

##=-=-=-=-=-=-=-=-=-=-=
# LTSA LLE embedding

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(
    n_neighbors=30,
    n_components=2,
    method="modified",  # the best parameter among others!!!
    neighbors_algorithm='ball_tree',
    random_state=0
)
X_lle = lle.fit_transform(X)
plot_embedding(X_lle, 'Modified LLE Embedding', 'fig_ltsa-lle.png')

##=-=-=-=-=-=-=-=-=-=-=
# Isomap embedding

from sklearn.manifold import Isomap

isomap = Isomap(n_neighbors=30, n_components=2)
X_isomap = isomap.fit_transform(X)
plot_embedding(X_isomap, 'Isomap Embedding', 'fig_isomap.png')

##=-=-=-=-=-=-=-=-=-=-=
# Linear Discriminant Analysis embedding

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

lda = LinearDiscriminantAnalysis(n_components=2)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_lda = lda.fit_transform(X, y_encoded)
plot_embedding(X_lda, 'Linear Discriminant Analysis Embedding', 'fig_lda.png')

##=-=-=-=-=-=-=-=-=-=-=
# Truncated SVD embedding

from sklearn.decomposition import TruncatedSVD

tsvd = TruncatedSVD(n_components=2)
X_tsvd = tsvd.fit_transform(X)
plot_embedding(X_tsvd, 'Truncated SVD Embedding', 'fig_svd.png')

##=-=-=-=-=-=-=-=-=-=-=
# Sparse Random projection embedding

from sklearn.random_projection import SparseRandomProjection

srp = SparseRandomProjection(
    n_components=2, density=0.3, eps=0.1, random_state=0,
    dense_output=False)
X_srp = srp.fit_transform(X)
plot_embedding(X_srp, 'Sparse Random projection Embedding', 'fig_srp.png')

##=-=-=-=-=-=-=-=-=-=-=
# PCA embedding

from sklearn.decomposition import PCA

pca = PCA(
    n_components=2,
    whiten=True,
    svd_solver='full',
    random_state=0,
    # copy='True',
    iterated_power=500
)
X_pca = pca.fit_transform(X)
plot_embedding(X_pca, 'PCA projection Embedding', 'fig_pca.png')

##=-=-=-=-=-=-=-=-=-=-=
# Kernel PCA embedding

from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# Initialize Kernel PCA with parameters
kpca = KernelPCA(
    n_components=2,
    kernel='poly',      # high and middle values
    gamma=0.3,
    degree=2,
    coef0=0.3,
    eigen_solver='auto',
    ## kernel='rbf',    # low values
    ## gamma=0.03,
    ## eigen_solver='auto',
    # kernel='sigmoid', # high and low values
    # gamma=0.04,
    # coef0=5.0,
    # eigen_solver='auto',
)
X_kpca = kpca.fit_transform(X)
plot_embedding(X_kpca, 'Kernel PCA Projection Embedding', 'fig_kpca.png')

##=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np
from openTSNE import TSNE
from openTSNE.initialization import pca
# from openTSNE.callbacks import ErrorLogger

ptsne = TSNE(
    initialization='pca',
    n_components=2,
    perplexity=5,
    random_state=0,
    early_exaggeration=5,
    metric='euclidean',
    verbose=False,
    theta=0.2
)
X_ptsne = ptsne.fit(X)
plot_embedding(X_ptsne, 'Parametric TSNE from OpenTSNE Embedding', 'fig_ptsne.png')

##=-=-=-=-=-=-=-=-=-=-=-=-=-=
plt.show()

##=-=-=-=-=-=-=-=-=-=-=-=-=
import matplotlib.pyplot as plt

# Create embeddings and corresponding figures
# fig, axs = plt.subplots(3, 4, figsize=(8, 6))
#
# axs[0,0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                edgecolor='black', linewidths=0.5)
# axs[0,0].set_title('t-SNE')
# axs[0,0].set_xlabel('Component 1')
# axs[0,0].set_ylabel('Component 2')
#
# axs[0,1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                edgecolor='black', linewidths=0.5)
# axs[0,1].set_title('UMAP')
# axs[0,1].set_xlabel('Component 1')
# axs[0,1].set_ylabel('Component 2')
#
# axs[0,2].scatter(X_spec[:, 0], X_spec[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                edgecolor='black', linewidths=0.5)
# axs[0,2].set_title('Spectral')
# axs[0,2].set_xlabel('Component 1')
# axs[0,2].set_ylabel('Component 2')
#
# axs[0,3].scatter(X_rt[:, 0], X_rt[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                edgecolor='black', linewidths=0.5)
# axs[0,3].set_title('Random Tree')
# axs[0,3].set_xlabel('Component 1')
# axs[0,3].set_ylabel('Component 2')
#
# axs[1,0].scatter(X_mds[:, 0], X_mds[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                edgecolor='black', linewidths=0.5)
# axs[1,0].set_title('MDS')
# axs[1,0].set_xlabel('Component 1')
# axs[1,0].set_ylabel('Component 2')
#
# axs[1,1].scatter(X_lle[:, 0], X_lle[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                edgecolor='black', linewidths=0.5)
# axs[1,1].set_title('LLE')
# axs[1,1].set_xlabel('Component 1')
# axs[1,1].set_ylabel('Component 2')
#
# axs[1,2].scatter(X_isomap[:, 0], X_isomap[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                edgecolor='black', linewidths=0.5)
# axs[1,2].set_title('Isomap')
# axs[1,2].set_xlabel('Component 1')
# axs[1,2].set_ylabel('Component 2')
#
# axs[1,3].scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                edgecolor='black', linewidths=0.5)
# axs[1,3].set_title('LDA')
# axs[1,3].set_xlabel('Component 1')
# axs[1,3].set_ylabel('Component 2')
#
# axs[2,0].scatter(X_tsvd[:, 0], X_tsvd[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                edgecolor='black', linewidths=0.5)
# axs[2,0].set_title('TSVD')
# axs[2,0].set_xlabel('Component 1')
# axs[2,0].set_ylabel('Component 2')
#
# axs[2,1].scatter(X_srp[:, 0], X_srp[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                  edgecolor='black', linewidths=0.5)
# axs[2,1].set_title('SPR')
# axs[2,1].set_xlabel('Component 1')
# axs[2,1].set_ylabel('Component 2')
#
# axs[2,2].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                  edgecolor='black', linewidths=0.5)
# axs[2,2].set_title('PCA')
# axs[2,2].set_xlabel('Component 1')
# axs[2,2].set_ylabel('Component 2')
#
# axs[2,3].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
#                  edgecolor='black', linewidths=0.5)
# axs[2,3].set_title('Kernel PCA')
# axs[2,3].set_xlabel('Component 1')
# axs[2,3].set_ylabel('Component 2')

plot_data = [
    (X_tsne, 't-SNE'),
    (X_umap, 'UMAP'),
    (X_spec, 'Spectral'),
    (X_rt, 'Random Tree'),
    (X_mds, 'MDS'),
    (X_lle, 'LLE'),
    (X_isomap, 'Isomap'),
    (X_lda, 'LDA'),
    (X_tsvd, 'TSVD'),
    (X_srp, 'SPR'),
    (X_pca, 'PCA'),
    (X_kpca, 'Kernel PCA')
]

fig, axs = plt.subplots(3, 4, figsize=(8, 6))

for i, (X, title) in enumerate(plot_data):
    row = i // 4
    col = i % 4

    ax = axs[row, col]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap('seismic', 12), s=20, alpha=0.90,
               edgecolor='black', linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

# =-=-=-=-=-=-=-=-=-=-=
plt.tight_layout()
plt.savefig(f'fig_tile.png')
plt.show()

##=-=-=-=-=-=-=-=-=-=-=
histogram_plot()
plt.savefig(f'fig_hist.png')
plt.show()

# Ensure final zero exit code
sys.exit(0)