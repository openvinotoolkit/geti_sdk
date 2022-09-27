import os
import tempfile
from typing import Union, Sequence

import cv2
import numpy as np

from .media_manager import BaseMediaManager
from sc_api_tools.data_models import MediaType, Video
from sc_api_tools.data_models.containers import MediaList
from sc_api_tools.rest_converters import MediaRESTConverter


class VideoManager(BaseMediaManager[Video]):
    """
    Class to manage video uploads and downloads for a certain project
    """

    _MEDIA_TYPE = MediaType.VIDEO

    def get_all_videos(self):
        """
        Get the ID's and filenames of all videos in the project

        :return: Dictionary containing the ID's (as keys) and filenames (as values)
            of the videos in the project
        """
        return self._get_all()

    def upload_video(self, video: Union[np.ndarray, str, os.PathLike]) -> Video:
        """
        Upload a video file to the server. Accepts either a path to a video file, or
        a numpy array containing pixel data for video frames.

        In case a numpy array is passed, this method expects the array to be 4
        dimensional, it's dimensions shaped as: [frames, heigth, width, channels]. The
        framerate of the created video will be set to 1 fps.

        :param video: full path to the video on disk, or numpy array holding the video
            pixel data
        :return: String containing the unique ID of the video, generated by Sonoma
            Creek
        """
        temporary_file_created = False
        if isinstance(video, (str, os.PathLike)):
            video_path = video
        elif isinstance(video, np.ndarray):
            try:
                n_frames, frame_height, frame_width, channels = video.shape
            except ValueError as error:
                raise ValueError(
                    f"Invalid video input shape, expected a 4D numpy array with "
                    f"dimensions representing [frames, height, width, channels]. Got "
                    f"shape {video.shape}"
                ) from error
            file_out = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
            out = cv2.VideoWriter(
                file_out.name,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1,
                (frame_width, frame_height)
            )
            for frame in video[:, ...]:
                out.write(frame)
            out.release()
            video_path = file_out.name
            temporary_file_created = True
        else:
            raise TypeError(f"Invalid video type: {type(video)}.")

        video_dict = self._upload(video_path)
        uploaded_video = MediaRESTConverter.from_dict(
            input_dict=video_dict, media_type=Video
        )
        uploaded_video._data = video_path
        uploaded_video._needs_tempfile_deletion = temporary_file_created
        return uploaded_video

    def upload_folder(
            self,
            path_to_folder: str,
            n_videos: int = -1,
            skip_if_filename_exists: bool = False
    ) -> MediaList[Video]:
        """
        Uploads all videos in a folder to the project. Returns the mapping of video
        filename to the unique ID assigned by Sonoma Creek.

        :param path_to_folder: Folder with videos to upload
        :param n_videos: Number of videos to upload from folder
        :param skip_if_filename_exists: Set to True to skip uploading of a video
            if a video with the same filename already exists in the project.
            Defaults to False
        :return: MediaList containing all video's in the project
        """
        return self._upload_folder(
            path_to_folder=path_to_folder,
            n_media=n_videos,
            skip_if_filename_exists=skip_if_filename_exists
        )

    def download_all(self, path_to_folder: str, append_video_uid: bool = False) -> None:
        """
        Download all videos in a project to a folder on the local disk.

        :param path_to_folder: path to the folder in which the videos should be saved
        :param append_video_uid: True to append the UID of a video to the
            filename (separated from the original filename by an underscore, i.e.
            '{filename}_{video_id}'). If there are videos in the project with
            duplicate filename, this must be set to True to ensure all videos are
            downloaded. Otherwise videos with the same name will be skipped.
        """
        self._download_all(path_to_folder, append_media_uid=append_video_uid)

    def delete_videos(self, videos: Sequence[Video]) -> bool:
        """
        Deletes all Video entities in `videos` from the project

        :param videos: List of Video entities to delete
        :return: True if all videos on the list were deleted successfully,
            False otherwise
        """
        return self._delete_media(media_list=videos)