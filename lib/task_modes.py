"""Bernini task-mode inference from connected inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BerniniTask(str, Enum):
    T2V = "t2v"
    V2V = "v2v"
    R2V = "r2v"
    RV2V = "rv2v"
    ADS2V = "ads2v"


TASK_DESCRIPTIONS = {
    BerniniTask.T2V: "Text-to-video (no visual context attached)",
    BerniniTask.V2V: "Video editing — source video only",
    BerniniTask.R2V: "Reference-to-video — reference images only",
    BerniniTask.RV2V: "Reference-guided video editing — source + references",
    BerniniTask.ADS2V: "Content insertion — source canvas + reference video",
}


@dataclass(frozen=True)
class TaskSummary:
    mode: BerniniTask
    stream_count: int
    has_source: bool
    has_ref_video: bool
    ref_image_count: int

    @property
    def label(self) -> str:
        return self.mode.value

    @property
    def description(self) -> str:
        return TASK_DESCRIPTIONS[self.mode]


def infer_task(
    has_source_video: bool,
    has_reference_video: bool,
    ref_image_count: int,
) -> BerniniTask:
    if has_source_video and has_reference_video:
        return BerniniTask.ADS2V
    if has_source_video and ref_image_count > 0:
        return BerniniTask.RV2V
    if has_source_video:
        return BerniniTask.V2V
    if ref_image_count > 0:
        return BerniniTask.R2V
    return BerniniTask.T2V
