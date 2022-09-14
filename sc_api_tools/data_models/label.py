# Copyright (C) 2022 Intel Corporation
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

import copy
from typing import ClassVar, List, Optional, Tuple

import attr
from ote_sdk.entities.color import Color
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel as OteScoredLabel

from sc_api_tools.data_models.enums import TaskType


@attr.s(auto_attribs=True)
class LabelSource:
    """
    Representation of a source for a ScoredLabel in SC

    :var user_id: ID of the user who assigned the label, if any
    :var model_id: ID of the model which generated the label, if any
    :var model_storage_id: ID of the model storage to which the model belongs
    """

    user_id: Optional[str] = None
    model_id: Optional[str] = None
    model_storage_id: Optional[str] = None
    # keys 'id' and 'type' are deprecated in v1.2, but required for v1.1
    id: Optional[str] = None
    type: Optional[str] = None


@attr.s(auto_attribs=True)
class Label:
    """
    Representation of a Label in SC.

    :var name: Name of the label
    :var id: Unique database ID of the label
    :var color: Color (in hex representation) of the label
    :var group: Name of the label group to which the label belongs
    :var is_empty: True if the label represents an empty label, False otherwise
    :var parent_id: Optional name of the parent label, if any
    """

    _identifier_fields: ClassVar[List[str]] = ["id", "hotkey"]

    name: str
    color: str
    group: str
    is_empty: bool
    hotkey: str = ""
    id: Optional[str] = None
    parent_id: Optional[str] = None
    is_anomalous: Optional[bool] = None

    def to_ote(self, task_type: TaskType) -> LabelEntity:
        """
        Convert the `Label` instance to an OTE SDK LabelEntity object.

        :return: OTE SDK LabelEntity instance corresponding to the label
        """
        return LabelEntity(
            name=self.name,
            domain=task_type.to_ote_domain(),
            id=self.id,
            hotkey=self.hotkey,
            is_empty=self.is_empty,
            color=Color.from_hex_str(self.color),
        )


@attr.s(auto_attribs=True)
class ScoredLabel:
    """
    Representation of a Label with an assigned probability in SC.

    :var name: Name of the label
    :var id: Unique database ID of the label
    :var color: Color (in hex representation) of the label
    :var probability:
    :var source:
    """

    _identifier_fields: ClassVar[List[str]] = ["id"]

    probability: float
    name: Optional[str] = None
    color: Optional[str] = None
    id: Optional[str] = None
    source: Optional[LabelSource] = None

    @property
    def color_tuple(self) -> Tuple[int, int, int]:
        """
        Return the color of the label as an RGB tuple.

        :return:
        """
        hex_color_str = copy.deepcopy(self.color).strip("#")
        return tuple(int(hex_color_str[i : i + 2], 16) for i in (0, 2, 4))

    @classmethod
    def from_label(cls, label: Label, probability: float) -> "ScoredLabel":
        """
        Create a ScoredLabel instance from an input Label and probability score.

        :param label: Label to convert to ScoredLabel
        :param probability: probability score for the label
        :return: ScoredLabel instance corresponding to `label` and `probability`
        """
        return ScoredLabel(
            name=label.name, probability=probability, color=label.color, id=label.id
        )

    @classmethod
    def from_ote(cls, ote_label: OteScoredLabel) -> "ScoredLabel":
        """
        Create a :py:class`~sc_api_tools.data_models.label.ScoredLabel` from
        the OTE SDK ScoredLabel entity passed.

        :param ote_label: OTE SDK ScoredLabel entity to convert from
        :return: ScoredLabel instance created according to the ote_label
        """
        return cls(
            name=ote_label.name,
            id=ote_label.id,
            probability=ote_label.probability,
            color=ote_label.color.hex_str,
        )