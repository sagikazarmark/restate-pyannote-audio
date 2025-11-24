from pathlib import Path

import torch
from pyannote.audio import Pipeline


class DefaultPipelineFactory:
    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-community-1",
        token: str | None = None,
        cache_dir: Path | str | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.token = token
        self.cache_dir = cache_dir
        self.device = device

    def create(self, model: str | None = None) -> Pipeline:
        if model is None:
            model = self.model

        pipeline = Pipeline.from_pretrained(
            model,
            token=self.token,
            cache_dir=self.cache_dir,
        )

        if pipeline is None:
            raise RuntimeError("Failed to create pipeline")

        if self.device is not None:
            pipeline = pipeline.to(self.device)

        return pipeline
