import logging
from typing import TYPE_CHECKING, Literal, cast

import fsspec
import fsspec.config
import obstore.fsspec
import pydantic_obstore
import restate
import torch
import workstate
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from obstore.store import ClientConfig

from restate_pyannote_audio import (
    DefaultPipelineFactory,
    Executor,
    create_service,
)


class ObstoreSettings(pydantic_obstore.Config):
    url: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")  # pyright: ignore[reportUnannotatedClassAttribute]

    model: str = "pyannote/speaker-diarization-community-1"
    token: str | None = None

    torch_device: str | None = None

    state_driver: Literal["fsspec"] | Literal["obstore"] = "fsspec"

    obstore: ObstoreSettings

    service_name: str = "pyannote-audio"

    identity_keys: list[str] = Field(alias="restate_identity_keys", default=[])


settings = Settings()  # pyright: ignore[reportCallIssue]

logging.basicConfig(level=logging.INFO)

if settings.state_driver == "fsspec":
    obstore.fsspec.register()

    for protocol in obstore.fsspec.SUPPORTED_PROTOCOLS - {"file", "memory"}:
        if settings.obstore.client_options is not None:
            fsspec.config.conf[protocol] = {
                "client_options": cast(
                    "ClientConfig | None",
                    settings.obstore.client_options.model_dump(exclude_none=True),
                ),
            }

    loader = workstate.fsspec.FileLoader()
    persister = workstate.fsspec.FilePersister()
else:
    store = (
        obstore.store.from_url(settings.obstore.url) if settings.obstore.url else None
    )
    client_options = cast(
        "ClientConfig | None",
        (
            settings.obstore.client_options.model_dump(exclude_none=True)
            if settings.obstore.client_options
            else None
        ),
    )

    loader = workstate.obstore.FileLoader(store, client_options=client_options)
    persister = workstate.obstore.FilePersister(store, client_options=client_options)

if settings.torch_device is None:
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    torch_device = torch.device(settings.torch_device)

pipeline_factory = DefaultPipelineFactory(
    settings.model,
    settings.token,
    device=torch_device,
    # todo: cache dir
)

executor = Executor(pipeline_factory, loader, persister)

service = create_service(executor, service_name=settings.service_name)

app = restate.app(services=[service], identity_keys=settings.identity_keys)
