from dataclasses import dataclass
from typing import Literal, Optional

IMAGE_SEG = "image_seg"
IMAGE_CLS = "image_cls"
VIDEO_SEG = "video_seg"
VIDEO_CLS = "video_cls"

ALL_TASKS = {
    IMAGE_SEG,
    IMAGE_CLS,
    VIDEO_SEG,
    VIDEO_CLS,
}

BUS_IMAGE = "bus_image"
BUS_VIDEO = "bus_video"
CEUS_VIDEO = "ceus_video"

ALL_MODALITIES = {
    BUS_IMAGE,
    BUS_VIDEO,
    CEUS_VIDEO,
}

POSITION_PROMPT = "position_prompt"
TASK_PROMPT = "task_prompt"
MODE_PROMPT = "mode_prompt"
TYPE_PROMPT = "type_prompt"

IMAGE = "image"
LABEL_CLS = "label_cls"
LABEL_SEG = "label_seg"
TASK_TYPE = "task_type"
MODALITY = "modality"
DATASET_NAME = "dataset_name"
IS_VIDEO = "is_video"
NUM_FRAMES = "num_frames"
PATIENT_ID = "patient_id"
CASE_ID = "case_id"

IGNORE_CLS_LABEL = -1
IGNORE_SEG_LABEL = None


@dataclass
class SampleSpec:
    image: object
    label_cls: int
    label_seg: object
    task_type: Literal["image_seg", "image_cls", "video_seg", "video_cls"]
    modality: Literal["bus_image", "bus_video", "ceus_video"]
    dataset_name: str
    is_video: bool
    num_frames: int
    patient_id: Optional[str] = None
    case_id: Optional[str] = None
    position_prompt: Optional[object] = None
    task_prompt: Optional[object] = None
    mode_prompt: Optional[object] = None
    type_prompt: Optional[object] = None
