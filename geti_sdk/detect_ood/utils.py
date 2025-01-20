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


import json
import os
from typing import List, Union

import albumentations
import cv2
import faiss
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from geti_sdk import Geti
from geti_sdk.data_models import Prediction
from geti_sdk.deployment import Deployment
from geti_sdk.rest_clients import ModelClient

from .ood_data import DistributionDataItem, DistributionDataItemPurpose

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


def count_images_in_directory(
    dir_path: str, include_subdirectories: bool = True
) -> int:
    """
    Count the number of images in the directory (including subdirectories)
    :param dir_path: Path to the directory containing images
    :param include_subdirectories: If True, images in subdirectories are also counted
    :return: Number of images in the directory
    """
    count = 0
    if not os.path.exists(dir_path):
        return count
    if include_subdirectories:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if os.path.splitext(file)[-1] in IMAGE_EXTENSIONS:
                    count += 1
    else:
        for file in os.listdir(dir_path):
            if os.path.splitext(file)[-1] in IMAGE_EXTENSIONS:
                count += 1
    return count


def infer_images_in_directory(dir_path: str, deployment: Deployment):
    """
    Infer all images in the directory (including subdirectories) using the deployment
    :param dir_path: Path to the directory containing images
    :param deployment: Geti Deployment object to use for inference
    :return: Generator yielding the image path, image as numpy array and the prediction for each image
    """
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.splitext(file)[-1] in IMAGE_EXTENSIONS:
                image_path = os.path.join(root, file)
                img_numpy, prediction = infer_image_on_deployment(
                    deployment=deployment, image_path=image_path, explain=True
                )
                yield image_path, img_numpy, prediction


def split_data(
    data: List[DistributionDataItem],
    stratified: bool,
    split_ratio: float,
    purposes: (DistributionDataItemPurpose, DistributionDataItemPurpose) = (
        DistributionDataItemPurpose.TRAIN,
        DistributionDataItemPurpose.TEST,
    ),
) -> (List[DistributionDataItem], List[DistributionDataItem]):
    """
    Split and assign the data into two sets - TRAIN and VAL/TEST
    :param data: List of DistributionDataItems to be assigned with Train and Test purposes.
    :param stratified: If True, the split is stratified based on the annotated labels.
    :param split_ratio: The fraction of data to be used for training. The remaining data is used for testing.
    :param purposes: Tuple of two DistributionDataItemPurpose values representing the purpose of the data.
    :return: Tuple of two lists containing the TRAIN and TEST data respectively.
    """
    if stratified:
        labels = [item.annotated_label for item in data]
        x_train, x_test, y_train, y_test = train_test_split(
            data,
            labels,
            train_size=split_ratio,
            stratify=labels,
            shuffle=True,
            random_state=42,
        )
    else:
        x_train, x_test = train_test_split(
            data, train_size=split_ratio, shuffle=True, random_state=42
        )

    for item in x_train:
        item.purpose = purposes[0]

    for item in x_test:
        item.purpose = purposes[1]

    return x_train, x_test


def load_annotations(annotation_file: str) -> Union[str, None]:
    """
    Read the annotations from the annotation file downloaded from Geti and returns the single label.
    Only to be used for classification tasks an image is annotated with a single label.
    :param annotation_file: Path to the annotation file
    :return: Annotated label for the image, if available, else None
    """
    if os.path.exists(annotation_file):
        with open(annotation_file, "r") as f:
            annotation = json.load(f)
            return annotation["annotations"][0]["labels"][0]["name"]
    return None


def image_to_distribution_data_item(
    deployment: Deployment, image_path: str, annotation_label: Union[str, None]
) -> DistributionDataItem:
    """
    Prepare the DistributionDataItem for the given image. Infers the image and extracts the feature vector.
    :param deployment: Geti Deployment object to use for inference
    :param image_path: Path to the image
    :param annotation_label: Annotated label for the image (optional)
    return: DistributionDataItem for the image
    """
    _, prediction = infer_image_on_deployment(
        deployment=deployment, image_path=image_path, explain=True
    )
    return DistributionDataItem(
        media_name=os.path.splitext(os.path.basename(image_path))[0],
        media_path=image_path,
        annotated_label=annotation_label,
        raw_prediction=prediction,
    )


def calculate_ood_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    """
    Evaluate the performance of  an OOD model using various metrics.
    :param y_true: Numpy array of true labels.
    :param y_pred_prob: Numpy array of predicted probabilities.
    :return: A dictionary containing the evaluation metrics including accuracy, AUROC, F1 score,
             TPR at 1% and 5% FPR, and the corresponding thresholds.
    """
    # Convert predicted probabilities into binary predictions with 0.5 threshold
    y_pred = (y_pred_prob > 0.5).astype(float)

    # Standard evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    au_roc = roc_auc_score(y_true, y_pred_prob)

    # Next, we calculate the precision-recall curve and the ROC curve to get the best threshold for F1 score
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_prob)
    fscores_at_thresholds = (2 * precision * recall) / (precision + recall)
    max_fscore_idx = np.argmax(fscores_at_thresholds)

    # Get the best threshold by F1 score, ensuring it is valid
    if len(thresholds_pr) > 1:
        best_threshold_fscore = thresholds_pr[max_fscore_idx]
        best_threshold_fscore = validate_cood_prediction_threshold(
            best_threshold_fscore
        )
    else:
        best_threshold_fscore = None

    # Calculate the TPR at 1% and 5% FPR.
    # TPR@1FPR is the metric used in COOD paper.
    # TPR@1%FPR indicates the model’s ability to correctly detect OOD data
    # while allowing only 1% of ID data (class 0) to be falsely classified as OOD.

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
    tpr_1_fpr = tpr[np.argmin(np.abs(fpr - 0.01))]
    tpr_5_fpr = tpr[np.argmin(np.abs(fpr - 0.05))]
    # Get the thresholds corresponding to 1% and 5% FPR, ensuring they are valid
    threshold_1_fpr = validate_cood_prediction_threshold(
        thresholds_roc[np.argmin(np.abs(fpr - 0.01))]
    )
    threshold_5_fpr = validate_cood_prediction_threshold(
        thresholds_roc[np.argmin(np.abs(fpr - 0.05))]
    )

    return {
        "accuracy": accuracy,
        "auroc": au_roc,
        "f1": f1,
        "tpr_1_fpr": tpr_1_fpr,
        "tpr_5_fpr": tpr_5_fpr,
        "threshold_fscore": best_threshold_fscore,
        "threshold_tpr_at_1_fpr": threshold_1_fpr,
        "threshold_tpr_at_5_fpr": threshold_5_fpr,
    }


def ood_metrics_to_string(metrics: dict) -> str:
    """
    Convert the OOD metrics dictionary to a string for logging.
    :param metrics: Dictionary containing the OOD metrics
    :return: A string containing the OOD metrics
    """
    metrics_to_long_name = {
        "accuracy": "Accuracy",
        "auroc": "AUROC",
        "f1": "F1 Score",
        "tpr_1_fpr": "TPR@1%FPR",
        "tpr_5_fpr": "TPR@5%FPR",
    }
    metrics_str = "\n".join(
        [
            f"{metrics_to_long_name[metric]}: {metrics[metric]:.4f}"
            for metric in metrics_to_long_name
            if metric in metrics
        ]
    )
    return metrics_str


def infer_image_on_deployment(
    deployment: Deployment, image_path: str, explain: bool = False
) -> (np.ndarray, Prediction):
    """
    Infer the image and get the prediction using the deployment
    :param deployment: Geti Deployment object to use for inference
    :param image_path: Path to the image
    :param explain: If True, prediction will contain the feature vector and saliency maps
    :return: Image as a numpy array and the prediction object
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if explain:
        # Note that a check to see if xai model is present in the deployment is not done.
        # If the model is not present, then feature_vector will be None
        return img_rgb, deployment.explain(image=img_rgb)
    else:
        return img_rgb, deployment.infer(image=img_rgb)


def validate_cood_prediction_threshold(threshold: float) -> Union[float, None]:
    """
    Validate the threshold to ensure it is not 0, inf, or 1.
    :param threshold: The threshold value to validate.
    :return: A valid threshold or None if the threshold is invalid.
    """
    if np.isinf(threshold) or threshold == 0 or threshold == 1:
        return None
    return threshold


def get_deployment_with_xai_head(geti: Geti, model_client: ModelClient) -> Deployment:
    """
    Get a deployment that has an optimised model with an XAI head. If there are multiple models with XAI heads,
    the model with the best performance is selected.
    :param geti: Geti instance pointing to the GETi server
    :param model_client: Modelclient instance pointing to the Project where at least one trained model is present
    :return: Deployment object with the optimised model with an XAI head.
    """
    # Check if there's at least one trained model in the project
    models = model_client.get_all_active_models()
    if len(models) == 0:
        raise ValueError(
            "No trained models were found in the project, please either "
            "train a model first or specify an algorithm to train."
        )

    # We need the model which has xai enabled - this allows us to get the feature vector from the model.
    model_with_xai_head = None

    # TODO[OOD] : More model properties can be used to determine "best" model (size, precision with respect to accuracy)
    max_model_performance = -1
    for model in models:
        for optimised_model in model.optimized_models:
            if optimised_model.has_xai_head:
                model_performance = optimised_model.performance.score
                if model_performance > max_model_performance:
                    model_with_xai_head = optimised_model
                    max_model_performance = model_performance

    if model_with_xai_head is None:
        raise ValueError(
            "No trained model with an XAI head was found in the project, "
            "please train a model with an XAI head first."
        )

    deployment = geti.deploy_project(
        project_name=model_client.project.name, models=[model_with_xai_head]
    )

    return deployment


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


def stratified_selection(
    x, y, fraction: float, min_samples_per_class: int = 3
) -> (List, List):
    """
    Sub sample (reduce) a dataset (x,y) by a provided fraction while maintaining the class distribution
    Note that this is to be use only for collection where each x (data point or sample) has only one y (label).

    :param x: Data points (samples)
    :param y: Labels
    :param fraction: Fraction of the dataset to keep.
    :param min_samples_per_class: Minimum number of samples to keep per class. Note that a very small value for
    "fraction" can sometimes make a class empty. To avoid this, we keep a minimum number of samples per class.
    :return: Indices of the data points to keep in the reduced split
    """
    selected_data_indices = []

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
        selected_data_indices.extend(selected_indices)

    return selected_data_indices


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


def perform_knn_indexing(
    feature_vectors: np.ndarray, use_gpu: bool = False
) -> faiss.IndexFlatL2:
    """
    Perform KNN indexing on the feature vectors using the FAISS library
    :param feature_vectors: Feature vectors to build the KNN index on
    :param use_gpu: Whether to use GPU for KNN indexing. Default is False
    :return: KNN search index object
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


def calculate_entropy_nearest_neighbours(
    train_labels: np.ndarray,
    nns_labels_for_test_fts: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Calculate the "entropy", a measure of how different the k nearest neighbours are for a sample.
    The value always range between [0,1] . A 0 indicates that all the k nearest neighbours belong to one class.
        Not a highly informative sample
    A value of 1 indicates that the sample has k different nearest neighbours (no sample belongs to same class).
        More informative sample in theory

    :param train_labels: labels of the annotated images
    :param nns_labels_for_test_fts: labels of the k nearest neighbours for each test feature
    :param k: number of nearest neighbours to consider
    :return: Entropy scores for each test feature
    """
    # preallocate
    neighbour_bin_count = np.zeros((nns_labels_for_test_fts.shape[0], k), dtype=int)
    for i in range(nns_labels_for_test_fts.shape[0]):
        nn_labels = train_labels[nns_labels_for_test_fts[i, :]]

        _, nn_bin_count = np.unique(
            nn_labels,
            return_index=False,
            return_inverse=False,
            return_counts=True,
        )
        neighbour_bin_count[i, : nn_bin_count.shape[0]] = nn_bin_count
        # No correction for all samples from a class being in a test sample's nearest neighbours
        # is done.

    # Calculate entropy
    # Note : This Entropy lies within [0,1]
    # A fully uncertain sample has entropy of 1 (bin count looks like [1,1,1,1,1,1,1,1,1,1])
    # A fully certain sample has entropy of 0 (bin count looks like [10,0,0,0,0,0,0,0,0,0])
    entropy_scores = stats.entropy(neighbour_bin_count, axis=1, base=k)
    return entropy_scores


class CutoutTransform:
    """
    Cutout transform to apply on images. This can be used for generating out of distribution (OOD) samples from in
    distribution (ID) samples.
    """

    def __init__(
        self,
        number_of_cutouts: int = 1,
        min_cutout_size: float = 0.5,
        max_cutout_size: float = 0.7,
    ):
        """
        :param number_of_cutouts: Number of cutouts to apply on the image
        :param min_cutout_size: Minimum size of the cutout
        :param max_cutout_size: Maximum size of the cutout
        """
        # TODO[OOD]: Add more advanced OOD transforms like perlin noise
        transform = albumentations.Compose(
            [
                albumentations.CoarseDropout(
                    max_holes=number_of_cutouts,
                    p=1,
                    hole_width_range=(min_cutout_size, max_cutout_size),
                    hole_height_range=(min_cutout_size, max_cutout_size),
                )
            ]
        )
        self.transform = transform

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the cutout transform on the image
        """
        return self.transform(image=image)["image"]
