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


from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from .ood_data import DistributionDataItem
from .utils import (
    calculate_entropy_nearest_neighbours,
    fit_pca_model,
    fre_score,
    perform_knn_indexing,
    perform_knn_search,
)


class OODSubModel(metaclass=ABCMeta):
    """
    Base class for OOD detection sub-models.
    """

    def __init__(self):
        self._is_trained = False

    @abstractmethod
    def train(self, distribution_data: List[DistributionDataItem]):
        """
        Train the OOD detection sub-model using a list in-distribution data items
        """
        raise NotImplementedError

    def __call__(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Check if the model is trained and call the forward method.
        """
        if not self._is_trained:
            raise ValueError(
                "Model is not trained. Please train the model first before calling."
            )
        return self.forward(data_items)

    @abstractmethod
    def forward(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Return the OOD score for the given data items.
        """
        raise NotImplementedError

    @property
    def is_trained(self) -> bool:
        """
        Return True if the model is trained.
        """
        return self._is_trained

    def __repr__(self):
        """
        Return a string representation of the OODSubModel.
        """
        return f"{self.__class__.__name__} (is_trained={self.is_trained})"


class KNNBasedOODModel(OODSubModel):
    """
    Model for OOD detection based on k-Nearest Neighbours (kNN) search in the feature space.
    The model calculates OOD scores based on distance to the nearest neighbours, entropy among the nearest neighbours,
    and EnWeDi (which combines distance and entropy).
    """

    # # TODO[OOD]: Add more features to the model
    # 1) distance to prototypical center
    # 2) ldof (to expensive ?)
    # 3) exact combination of entropy and distance from thesis

    def __init__(self, knn_k: int = 10):
        super().__init__()
        self.knn_k = knn_k
        self.knn_search_index = None
        self.train_set_labels = None

    def train(self, distribution_data: List[DistributionDataItem]):
        """
        Build the knn search index using faiss for the in-distribution data.
        :param distribution_data: List of DistributionDataItems for training the model. These are typically user
        annotated images from a Geti project from datasets that correspond to "in-distribution". Please note that the
        annotated labels are required if entropy based ood scores measures are calculated in the forward method.
        """
        id_data = distribution_data
        feature_vectors = np.array([data.feature_vector for data in id_data])
        labeled_set_labels = np.array([data.annotated_label for data in id_data])

        self.train_set_labels = labeled_set_labels
        self.knn_search_index = perform_knn_indexing(feature_vectors, use_gpu=False)
        self._is_trained = True

    def forward(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Perform kNN search and calculates different types of OOD scores.
        :param data_items: List of DistributionDataItems for which OOD scores are calculated
        :return: A dictionary containing the OOD score names as keys and the OOD scores as values.
        """
        features = np.array([item.feature_vector for item in data_items])
        distances, nn_indices = perform_knn_search(
            knn_search_index=self.knn_search_index,
            feature_vectors=features,
            k=self.knn_k,
        )

        knn_distance = distances[:, -1]  # distance to the kth nearest neighbour
        nn_distance = distances[:, 1]  # distance to the nearest neighbour
        # TODO[OOD] : When doing kNN Search for ID, the 0th index is the same image. So, should we use k+1 ?
        average_nn_distance = np.mean(distances[:, 1:], axis=1)

        entropy_score = calculate_entropy_nearest_neighbours(
            train_labels=self.train_set_labels,
            nns_labels_for_test_fts=nn_indices,
            k=self.knn_k,
        )

        # Add one to the entropy scores
        # This is to offset the range to [1,2] instead of [0,1] and avoids division by zero
        # if used elsewhere
        entropy_score += 1

        enwedi_score = average_nn_distance * entropy_score
        enwedi_nn_score = nn_distance * entropy_score

        return {
            "knn_distance": knn_distance,
            "nn_distance": nn_distance,
            "average_nn_distance": average_nn_distance,
            "entropy_score": entropy_score,
            "enwedi_score": enwedi_score,
            "enwedi_nn_score": enwedi_nn_score,
        }


class GlobalFREBasedModel(OODSubModel):
    """
    Global Feature Reconstruction Error (FRE) Model. Builds a single PCA model for the whole in-distribution
    data provided thereby providing a "Global" subspace representation of the in-distribution features.
    See https://arxiv.org/abs/2012.04250 for details.
    """

    def __init__(self, n_components=0.995):
        super().__init__()
        self.n_components = n_components
        self.pca_model = None

    def train(self, distribution_data: List[DistributionDataItem]):
        """
        Fit a single (global) PCA model for the in-distribution data
        """
        feature_vectors = np.array([data.feature_vector for data in distribution_data])
        self.pca_model = fit_pca_model(
            feature_vectors=feature_vectors, n_components=self.n_components
        )
        self._is_trained = True

    def forward(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Return the global fre score for the given data items.
        """
        if not self._is_trained:
            raise ValueError(
                "Model is not trained. Please train the model first before calling."
            )
        features = np.array([item.feature_vector for item in data_items])
        fre_scores = fre_score(feature_vectors=features, pca_model=self.pca_model)
        return {"global_fre_score": fre_scores}


class ClassFREBasedModel(OODSubModel):
    """
    Per-class Feature Reconstruction Error (FRE) Model. Each class present in the in-distribution data is represented
    by a subspace model.
    See https://arxiv.org/abs/2012.04250 for details
    """

    def __init__(self, n_components=0.995):
        super().__init__()
        self.n_components = n_components
        self.pca_models_per_class = {}

    def train(self, distribution_data: List[DistributionDataItem]):
        """
        Fit PCA Models on the in-distribution data for each class.
        """
        id_data = distribution_data
        feature_vectors = np.array([data.feature_vector for data in id_data])
        labels = np.array([data.annotated_label for data in id_data])

        # iterate through unique labels and fit pca model for each class
        pca_models = {}

        for label in np.unique(labels):
            # labels are list of class names and not indices
            class_indices = [i for i, j in enumerate(labels) if j == label]
            class_features = feature_vectors[class_indices]
            pca_models[label] = fit_pca_model(
                feature_vectors=class_features, n_components=self.n_components
            )

        self.pca_models_per_class = pca_models
        self._is_trained = True

    def forward(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Return various fre-based ood scores
        """
        if not self._is_trained:
            raise ValueError(
                "Model is not trained. Please train the model first before calling."
            )

        num_data_items = len(data_items)
        features = np.array([item.feature_vector for item in data_items])
        fre_scores_per_class = {}
        # class_fre_models is a dict with label name and pca model.
        for label, pca_model in self.pca_models_per_class.items():
            fre_scores_per_class[label] = fre_score(
                feature_vectors=features, pca_model=pca_model
            )

        # FRE Score # 1 - FRE  w.r.t the class the sample is predicted to be
        predicted_labels = [item.predicted_label for item in data_items]
        fre_scores_for_predicted_class = np.array(
            [fre_scores_per_class[label][i] for i, label in enumerate(predicted_labels)]
        )

        # FRE Score # 2 - Calculating the minimum FRE score across all classes

        min_fre_scores = np.zeros(num_data_items)
        # For each data point, find the minimum FRE score across all classes (labels)
        for i in range(num_data_items):
            min_fre_scores[i] = np.min(
                [fre_scores_per_class[label][i] for label in fre_scores_per_class]
            )

        # Note - It is observed that the minimum FRE scores are almost always same as the FRE scores for the predicted
        # class i.e., the predicted class is the class with minimum fre score.
        # However, this is true largely for ID images (99.95% of example data points).
        # For OOD images, this applies, but less frequently (78.3% of example data points).
        # Therefore, the difference of the two scores can also be considered as a "feature"

        return {
            "min_class_fre_score": min_fre_scores,
            "predicted_class_fre_score": fre_scores_for_predicted_class,
            "diff_min_and_predicted_class_fre": 1e-8
            + (fre_scores_for_predicted_class - min_fre_scores),
        }


class ProbabilityBasedModel(OODSubModel):
    """
    Maximum Softmax Probability Model - A baseline OOD detection model.
    Uses the concept that a lower maximum softmax probability indicates that the image could be OOD.
    """

    def __init__(self):
        super().__init__()

    def train(self, distribution_data: List[DistributionDataItem]):
        """
        MSP model does not require training.
        """
        self._is_trained = True

    def forward(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Return the maximum softmax probability for the given prediction.
        """
        msp_scores = np.ndarray(len(data_items))
        for i, data_item in enumerate(data_items):
            # deployment.infer gives a single highest probability- no need to find the max
            msp_scores[i] = data_item.max_prediction_probability[0]

        return {"max_softmax_probability": msp_scores}
