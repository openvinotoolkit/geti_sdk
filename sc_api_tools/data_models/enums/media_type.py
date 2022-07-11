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

from enum import Enum


class MediaType(Enum):
    """
    Enum representing the different media types in SC.
    """

    IMAGE = 'image'
    VIDEO = 'video'
    VIDEO_FRAME = 'video_frame'

    def __str__(self) -> str:
        """
        Return the string representation of the MediaType instance.
        """
        return self.value


SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mkv", ".mov"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"]