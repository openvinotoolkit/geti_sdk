# Copyright (C) 2023 Intel Corporation
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
Introduction
------------

The "detect_ood" package contains the
:py:class:`~geti_sdk.detect_ood.OODModel` class, which provides
Out-of-distribution detection functions (training an OODModel as well as detecting OOD samples).

Primarily, it is used by the OODTrigger class (~geti_sdk.post_inference_hooks.triggers.ood_trigger.OODTrigger)
to detect out-of-distribution samples.

Module contents
---------------

.. automodule:: geti_sdk.detect_ood.OODModel
   :members:
   :undoc-members:
   :show-inheritance:
"""

from .ood_model import COODModel

__all__ = ["COODModel"]
