from typing import Any, Optional, Union

from starlette.responses import Response
from starlette.requests import HTTPConnection, Request
from starlette.types import Message

from starlette_context.plugins.base import PluginUUIDBase
from starlette_context import context


class SessionPlugin(PluginUUIDBase):
    # The returned value will be inserted in the context with this key
    key = "session_cookie"
    cookie_name = "haystack-session"

    async def process_request(
        self, request: Union[Request, HTTPConnection]
    ) -> Optional[Any]:
        session_cookie = request.cookies.get(self.cookie_name)
        if not session_cookie:
            session_cookie = self.get_new_uuid()
        return session_cookie

    async def enrich_response(self, response: Union[Response, Message]) -> None:
        session_cookie = str(context.get(self.key))
        if isinstance(response, Response):
            response.set_cookie(self.cookie_name, session_cookie)
