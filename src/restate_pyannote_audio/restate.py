import restate

from . import DiarizeRequest, DiarizeResponse, Executor


def create_service(
    executor: Executor,
    service_name: str = "pyannote-audio",
) -> restate.Service:
    service = restate.Service(service_name)

    register_service(executor, service)

    return service


def register_service(
    executor: Executor,
    service: restate.Service,
):
    @service.handler()
    async def diarize(
        ctx: restate.Context,
        request: DiarizeRequest,
    ) -> DiarizeResponse:
        return await ctx.run_typed(
            "diarize",
            executor.diarize,
            request=request,
        )
