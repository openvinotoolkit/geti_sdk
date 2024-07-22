# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


import faiss
import numpy as np
from sklearn.decomposition import PCA

from geti_sdk.deployment import Deployment
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_clients import ImageClient


def fit_pca_model(feature_vectors=np.ndarray, n_components: float = 0.995) -> PCA:
    """
    Fit a Principal component analysis (PCA) model to the features and returns the model
    :param feature_vectors: Train set features to fit the PCA model
    :param n_components: Number of components (fraction of variance) to keep
    :return: A fitted PCA model
    """
    pca_model = PCA(n_components)
    pca_model.fit(feature_vectors)
    return pca_model


def stratified_selection(x, y, fraction: float, min_samples_per_class: int = 3):
    """
    Sub sample (reduce) a dataset (x,y) by a provided fraction while maintaining the class distribution
    Note that this is to be use only for collection where each x (data point or sample) has only one y (label).

    :param x: Data points (samples)
    :param y: Labels
    :param fraction: Fraction of the dataset to keep.
    :param min_samples_per_class: Minimum number of samples to keep per class. Note that a very small value for
    "fraction" can sometimes make a class empty. To avoid this, we keep a minimum number of samples per class.
    """
    # TODO[ood]: Yet to be tested
    # stratified sampling from train_labels

    selected_labels = []
    selected_features = []

    samples = x
    labels = y

    # Check if labels is empty
    if len(labels) == 0:
        raise ValueError("Labels cannot be empty")

    # Check if len of labels and samples are equal
    if len(labels) != len(samples):
        raise ValueError("Length of labels and samples must be equal")

    if type(labels) is list:
        labels = np.array(labels)

    # Get unique labels
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        # Get number of samples to keep
        n_samples_to_keep = max(
            min_samples_per_class, int(fraction * len(label_indices))
        )
        selected_indices = np.random.choice(
            label_indices, n_samples_to_keep, replace=False
        )
        # Append selected samples and labels
        selected_labels.extend(labels[selected_indices])
        selected_features.extend(samples[selected_indices])

    return selected_features, selected_labels


def fre_score(feature_vectors: np.ndarray, pca_model: PCA) -> np.ndarray:
    """
    Calculate the feature reconstruction error (FRE) score for the given feature vector(s)
    :param feature_vectors: feature vectors to compute the FRE score
    :param pca_model: PCA model to use for computing the FRE score. PCA model must be fitted already
    :return: FRE scores for the given feature vectors
    """
    features_original = feature_vectors
    features_transformed = pca_model.transform(feature_vectors)
    features_reconstructed = pca_model.inverse_transform(features_transformed)
    fre_scores = np.sum(np.square(features_original - features_reconstructed), axis=1)
    return fre_scores


def perform_knn_indexing(feature_vectors: np.ndarray, use_gpu: bool = False):
    """
    Perform KNN indexing on the feature vectors
    """
    # use faiss with gpu
    if use_gpu:
        res = faiss.StandardGpuResources()
        # build a flat (CPU) index
        index_flat = faiss.IndexFlatL2(feature_vectors.shape[1])
        # make it into a gpu index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(feature_vectors)
        return gpu_index_flat
    else:
        index_flat = faiss.IndexFlatL2(feature_vectors.shape[1])
        index_flat.add(feature_vectors)
        return index_flat


def perform_knn_search(
    knn_search_index: faiss.IndexFlatL2, feature_vectors: np.ndarray, k: int = 10
) -> (np.ndarray, np.ndarray):
    """
    Perform KNN search on the feature vectors in the feature space indexed by the knn_search_index
    :param knn_search_index: KNN search index. An object representing the indexed knn search space.
    Ideally this object is returned by perform_knn_indexing().
    :param feature_vectors: Query feature vectors to search in the indexed feature space.
        Note that the feature_vectors' size should be (N, M) where N is the number of feature vectors
        and M is the dimension of the feature vectors.
    :param k: Number of nearest neighbours to search for
    :return: distances, indices each of size (N,K). Note that distances are squared Euclidean distances.
    """
    distances, indices = knn_search_index.search(feature_vectors, k)

    return distances, indices


def normalise_features(feature_vectors: np.ndarray) -> np.ndarray:
    """
    Feature embeddings are normalised by dividing each feature embedding vector by its respective 2nd-order vector norm
    (vector Euclidean norm). It has been shown that normalising feature embeddings lead to a significant improvement
    in OOD detection.
    :param feature_vectors: Feature vectors to normalise
    :return: Normalised feature vectors.
    """
    return feature_vectors / (
        np.linalg.norm(feature_vectors, axis=1, keepdims=True) + 1e-10
    )


def extract_features_from_imageclient(
    deployment: Deployment,
    image_client: ImageClient,
    geti_session: GetiSession,
    n_images: int = -1,
    normalise_feats: bool = True,
):
    """
    Extract
    """
    pass


def generate_ood_dataset_by_corruption(
    geti_deployment: Deployment,
    source_path: str,
    corruption_type: str,
    dest_path: str = None,
    desired_accuracy: float = 50,
    desired_accuracy_tol=3.0,
    show_progress: bool = True,
) -> str:
    """
    Util
    """
    pass
