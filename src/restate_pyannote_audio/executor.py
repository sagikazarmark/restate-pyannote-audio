import tempfile
from typing import IO, Protocol

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from pydantic import AnyUrl, BaseModel, ConfigDict, Field
from restate.exceptions import TerminalError

# from pyannote.audio.pipelines.utils.hook import Hooks, ProgressHook


class PipelineFactory(Protocol):
    def create(self, model: str | None = None) -> Pipeline: ...


class Loader(Protocol):
    def load(self, ref: AnyUrl, dst: IO): ...


class Persister(Protocol):
    def persist(self, ref: AnyUrl, src: bytes | bytearray | memoryview): ...


class DiarizeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"inputRef": "s3://bucket/audio.wav"},
            ]
        }
    )

    inputRef: AnyUrl = Field(
        description="Input audio reference",
        examples=["s3://bucket/audio.wav", "https://example.com/audio.mp3"],
    )
    model: str | None = None
    outputRef: AnyUrl | None = Field(
        default=None,
        description="Output reference for diarization results",
        examples=["s3://bucket/results.json", "https://example.com/results.json"],
    )
    forceOutput: bool = Field(
        default=False,
        description="Force output even when output reference is set",
    )


class SpeechTurn(BaseModel):
    """Represents a single speech turn with timing and speaker info"""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"start": 0.0, "end": 1.0, "speaker": "speaker1"},
            ]
        }
    )

    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    speaker: str = Field(description="Speaker identifier")


class DiarizeResponse(BaseModel):
    diarization: list[SpeechTurn] = Field(description="Speaker diarization segments")
    exclusive_diarization: list[SpeechTurn] = Field(
        description="Non-overlapping speaker diarization segments"
    )

    @staticmethod
    def from_output(output: DiarizeOutput) -> "DiarizeResponse":
        """Convert dataclass DiarizeOutput to Pydantic DiarizeOutput

        Args:
            dataclass_instance: Instance of the dataclass DiarizeOutput

        Returns:
            Pydantic DiarizeOutput model
        """

        def _serialize(diarization) -> list[SpeechTurn]:
            turns = []
            for speech_turn, _, speaker in diarization.itertracks(yield_label=True):  # pyright: ignore[reportAssignmentType]
                turns.append(
                    SpeechTurn(
                        start=round(speech_turn.start, 3),
                        end=round(speech_turn.end, 3),
                        speaker=str(speaker),
                    )
                )

            return turns

        return DiarizeResponse(
            diarization=_serialize(output.speaker_diarization),
            exclusive_diarization=_serialize(output.exclusive_speaker_diarization),
        )


class Executor:
    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        loader: Loader,
        persister: Persister,
        debug: bool = False,
    ):
        self.pipeline_factory = pipeline_factory
        self.loader = loader
        self.persister = persister
        self.debug = debug

    def diarize(self, request: DiarizeRequest) -> DiarizeResponse:
        try:
            pipeline = self.pipeline_factory.create(request.model)
        except Exception as e:
            raise TerminalError(str(e))

        with tempfile.NamedTemporaryFile(delete=True) as file:
            self.loader.load(request.inputRef, dst=file)

            # TODO: use hooks locally when possible
            # hooks = []

            # if self.debug:
            #     hooks.append(ProgressHook())

            # with Hooks(*hooks) as hook:
            output: DiarizeOutput = pipeline(file.name)

        response = DiarizeResponse.from_output(output)

        if request.outputRef is not None:
            self.persister.persist(
                request.outputRef,
                response.model_dump_json(indent=4).encode(),
            )

            if not request.forceOutput:
                response = DiarizeResponse(
                    diarization=[],
                    exclusive_diarization=[],
                )

        return response
