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


import numpy as np

from geti_sdk.data_models import Prediction
from geti_sdk.deployment.inference_hook_interfaces import PostInferenceTrigger
from geti_sdk.detect_ood import COODModel


class OODTrigger(PostInferenceTrigger):
    """
    Post inference trigger based on the out-of-distribution (OOD) detection score for an image.
    A threshold is already provided by the OODModel already to determine if the image is OOD or not. If the OOD score
    is **higher** than the defined `threshold`, the trigger is activated.

    :param ood_model: OODModel object that calculates the OOD score for an image
    """

    def __init__(self, ood_model: COODModel, threshold: float = 0.5):
        super().__init__(threshold=threshold)
        self.ood_model = ood_model

        self._repr_info_ += ""

    def __call__(self, image: np.ndarray, prediction: Prediction) -> float:
        """
         Compute an OOD score for the 'image' using the corresponding information (feature_vector,prediction probabilities)
          from "prediction"

        :param image: Numpy array representing an image
        :param prediction: Prediction object corresponding to the inference result for the image.
        :return: Float representing the score for the input
        """
        cood_score = self.ood_model(prediction=prediction)
        return cood_score

    def get_decision(self, score: float) -> bool:
        """
        Make a decision to classify the sample into "in-distribution" or "out-of-distribution" based on
        the OOD score and threshold set for trigger.

        :param score: Float representing the OOD-score for an image.
        :return: Boolean indicating whether the trigger conditions are met (True,
            score is **higher** than the threshold), or not (False)
        """
        return score > self.threshold
