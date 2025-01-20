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

from enum import Enum
from typing import Union

import numpy as np

from geti_sdk.data_models import Prediction


class DistributionDataItemPurpose(Enum):
    """
    Enum to represent the purpose of the DistributionDataItem.
    This is used during splitting of the data into TRAIN, VAL, TEST
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DistributionDataItem:
    """
    A class to store the data for the COOD model.
    An DistributionDataItem for an image contains the following:
    - media_name: Name of the media (optional)
    - image_path: Path to the image (optional)
    - annotated_label: Annotated label for the image (optional)
    - raw_prediction: Prediction object for the image (required)
    - feature_vector: Feature vector extracted from the image (extracted from raw_prediction)

    All OOD models take a list of DistributionDataItems as input for training and inference.
    """

    def __init__(
        self,
        raw_prediction: Prediction,
        media_name: Union[str, None],
        media_path: Union[str, None],
        annotated_label: Union[str, None],
        normalise_feature_vector: bool = True,
        purpose: Union[DistributionDataItemPurpose, None] = None,
    ):
        self.media_name = media_name
        self.image_path = media_path
        self.annotated_label = annotated_label
        self.raw_prediction = raw_prediction
        self.purpose = purpose

        feature_vector = raw_prediction.feature_vector

        if len(feature_vector.shape) != 1:
            feature_vector = feature_vector.flatten()

        if normalise_feature_vector:
            feature_vector = self.normalise_features(feature_vector)[0]

        self._normalise_feature_vector = normalise_feature_vector
        self.feature_vector = feature_vector
        self.max_prediction_probability = (
            raw_prediction.annotations[0].labels[0].probability,
        )
        self.predicted_label = raw_prediction.annotations[0].labels[0].name

    @property
    def is_feature_vector_normalised(self) -> bool:
        """
        Return True if the feature vector is normalised.
        """
        return self._normalise_feature_vector

    @staticmethod
    def normalise_features(feature_vectors: np.ndarray) -> np.ndarray:
        """
        Feature embeddings are normalised by dividing each feature embedding vector by its respective 2nd-order vector
        norm (vector Euclidean norm). It has been shown that normalising feature embeddings lead to a significant
        improvement in OOD detection.
        :param feature_vectors: Feature vectors to normalise
        :return: Normalised feature vectors.
        """
        if len(feature_vectors.shape) == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        return feature_vectors / (
            np.linalg.norm(feature_vectors, axis=1, keepdims=True) + 1e-10
        )

    def __repr__(self):
        """
        Return a string representation of the DistributionDataItem.
        """
        return (
            f"DataItem(media_name={self.media_name}, "
            f"shape(feature_vector)={self.feature_vector.shape}), "
            f"feature_vector normalised={self.is_feature_vector_normalised})"
        )
