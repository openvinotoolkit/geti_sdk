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
"""Triggers that can be used in post inference hooks"""
from .always_trigger import AlwaysTrigger
from .confidence_trigger import ConfidenceTrigger
from .empty_label_trigger import EmptyLabelTrigger
from .label_trigger import LabelTrigger
from .object_count_trigger import ObjectCountTrigger
from .ood_trigger import OODTrigger

__all__ = [
    "AlwaysTrigger",
    "ConfidenceTrigger",
    "LabelTrigger",
    "EmptyLabelTrigger",
    "ObjectCountTrigger",
    "OODTrigger",
]
