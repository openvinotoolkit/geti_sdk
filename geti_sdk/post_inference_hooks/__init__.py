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
"""
PostInferenceHooks are used to construct pipelines for e.g. data collection or other
actions based on inference outcomes.
"""
from .actions import FileSystemDataCollection, GetiDataCollection, HttpRequestAction
from .post_inference_hook import PostInferenceHook
from .triggers import (
    AlwaysTrigger,
    ConfidenceTrigger,
    EmptyLabelTrigger,
    LabelTrigger,
    ObjectCountTrigger,
    OODTrigger,
)

__all__ = [
    "PostInferenceHook",
    "ConfidenceTrigger",
    "LabelTrigger",
    "EmptyLabelTrigger",
    "ObjectCountTrigger",
    "AlwaysTrigger",
    "OODTrigger",
    "GetiDataCollection",
    "FileSystemDataCollection",
    "HttpRequestAction",
]
