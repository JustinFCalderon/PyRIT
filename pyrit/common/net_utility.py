# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Literal, Optional, overload

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

PostType = Literal["json", "data"]


@overload
def get_httpx_client(
    *,
    use_async: Literal[True],
    debug: bool = False,
    **httpx_client_kwargs: Any,
) -> httpx.AsyncClient: ...


@overload
def get_httpx_client(
    *,
    use_async: Literal[False] = False,
    debug: bool = False,
    **httpx_client_kwargs: Any,
) -> httpx.Client: ...


def get_httpx_client(
    *,
    use_async: bool = False,
    debug: bool = False,
    **httpx_client_kwargs: Any,
) -> httpx.Client | httpx.AsyncClient:
    """
    Build an httpx Client/AsyncClient with sensible defaults for PyRIT.

    Notes:
    - httpx default timeout is short; we increase read timeout substantially.
    - `debug=True` routes through a local proxy (e.g., Fiddler/Burp) and disables TLS verify by default.
    """

    client_class = httpx.AsyncClient if use_async else httpx.Client

    # Keep PyRIT's existing behavior: use a local proxy when debug is enabled.
    default_proxy = "http://localhost:8080" if debug else None

    # Support either "proxy" (legacy in your code) or "proxies" (httpx kwarg)
    proxy = httpx_client_kwargs.pop("proxy", None)
    proxies = httpx_client_kwargs.pop("proxies", None)

    if proxy is None and proxies is None:
        proxies = default_proxy
    elif proxies is None:
        proxies = proxy

    verify_certs = httpx_client_kwargs.pop("verify", not debug)

    # httpx defaults are small; long generations frequently hit read timeouts.
    timeout = httpx_client_kwargs.pop(
        "timeout",
        httpx.Timeout(connect=300.0, read=1800.0, write=30.0, pool=300.0),
    )

    limits = httpx_client_kwargs.pop(
        "limits",
        httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )

    # Only pass proxies if we actually have one (None can be passed too, but keep it tidy)
    if proxies is not None:
        httpx_client_kwargs["proxy"] = proxies

    return client_class(
        verify=verify_certs,
        timeout=timeout,
        limits=limits,
        **httpx_client_kwargs,
    )


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=30),
    reraise=True,
)
async def make_request_and_raise_if_error_async(
    endpoint_uri: str,
    method: str,
    post_type: PostType = "json",
    debug: bool = False,
    params: Optional[dict[str, str]] = None,
    request_body: Optional[dict[str, object]] = None,
    headers: Optional[dict[str, str]] = None,
    **httpx_client_kwargs: Any,
) -> httpx.Response:
    """
    Make an async HTTP request and raise a helpful exception on failure.

    - Retries (tenacity) on httpx.TimeoutException / httpx.TransportError
    - Raises RuntimeError with a response preview on 4xx/5xx
    - Returns httpx.Response on success
    """
    headers = headers or {}
    params = params or {}

    async with get_httpx_client(debug=debug, use_async=True, **httpx_client_kwargs) as async_client:
        try:
            response = await async_client.request(
                method=method,
                url=endpoint_uri,
                params=params,
                json=request_body if (request_body is not None and post_type == "json") else None,
                data=request_body if (request_body is not None and post_type != "json") else None,
                headers=headers,
            )
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"HTTP timeout calling {method} {endpoint_uri} (timeout={async_client.timeout})"
            ) from e
        except httpx.TransportError as e:
            raise RuntimeError(
                f"HTTP transport error calling {method} {endpoint_uri}: {e!r}"
            ) from e

        if response.is_error:
            try:
                body_preview = response.text[:800]
            except Exception:
                body_preview = "<unable to read body>"
            raise RuntimeError(
                f"HTTP {response.status_code} for {method} {endpoint_uri}\n"
                f"Response preview:\n{body_preview}"
            )

        return response
