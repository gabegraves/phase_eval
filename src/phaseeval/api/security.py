from starlette.types import ASGIApp, Message, Receive, Scope, Send

from phaseeval.api.errors import problem_response
from phaseeval.kernel.errors import TooLarge


class BodyLimit:
    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            for name, value in scope["headers"]:
                if name == b"content-length":
                    try:
                        length = int(value)
                    except ValueError:
                        continue
                    if length > self.limit:
                        raise TooLarge()
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self.limit:
                    raise TooLarge()
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
        except TooLarge as error:
            await problem_response(error)(scope, receive, send)
            return

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)
