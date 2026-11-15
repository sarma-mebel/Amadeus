"""FastAPI-powered web application for the Amadeus assistant.

The heavy lifting lives in sibling modules — ``models`` (pydantic shapes),
``sessions`` (WebChatSession/SessionStore), and the ``routes_*`` router
factories. This module keeps app assembly (create_app), the optional bearer
auth middleware, static/SPA serving, and the module-level ``app`` used by
serving entrypoints. Moved names are re-exported below so existing imports
and test monkeypatches of ``amadeus.web.app.*`` keep working.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..logging_utils import get_logger, setup_logging
from ..paths import migrate_legacy_cwd_files

# Provider loaders are bound here (not only in provider_loader) on purpose:
# the test suite monkeypatches e.g. ``amadeus.web.app.load_tts``, and
# sessions.py resolves the loaders through this module at call time so those
# patches take effect even though SessionStore moved out of this file.
from ..provider_loader import (  # noqa: F401 - re-exported for tests/back-compat
    load_embedder,
    load_llm,
    load_stt,
    load_tts,
)
from . import (
    routes_chat,
    routes_characters,
    routes_meta,
    routes_persona,
    routes_sessions,
)
from .ratelimit import TurnRateLimiter, client_key

# Re-exports: these names used to be defined in this module.
from .models import (  # noqa: F401 - re-exported for tests/back-compat
    ChatRequest,
    ChatResponse,
    GreetingRequest,
    MemoryRequest,
    NewSessionRequest,
    PresenceRequest,
    ResetRequest,
    SentencePayload,
    TTSRequest,
)
from .blocking import (  # noqa: F401 - re-exported for tests/back-compat
    cached_translate as _cached_translate,
)
from .blocking import (  # noqa: F401
    run_blocking as _run_blocking,
)
from .sessions import (  # noqa: F401 - re-exported for tests/back-compat
    TRANSLATOR_AVAILABLE,
    ProviderCallError,
    SessionStore,
    WebChatSession,
    _synthesize_tts,
    _warn_if_llm_degraded,
)

LOGGER = get_logger(__name__)


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create and configure a FastAPI app instance."""

    if config is None:
        config_path = os.getenv("AMADEUS_CONFIG")
        config = Config.load(Path(config_path) if config_path else None)
    setup_logging(config)
    migrate_legacy_cwd_files(config)
    store = SessionStore(config)
    _warn_if_llm_degraded(config, store.llm)

    # Background scheduler (daily brief / diary / reach-out / follow-through /
    # nightly self-eval) rides the app lifespan: started only when at least
    # one job is enabled, cancelled cleanly on shutdown. Tests using
    # httpx.ASGITransport never execute lifespan, so they are unaffected.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import asyncio

        scheduler = None
        # Captured so the settings-reload path (settings_api.persist_and_
        # reload) can start a scheduler AFTER boot when a save newly enables
        # a scheduler feature — this gate runs once at startup, and without
        # a handle to the loop a later enable silently never ran anything
        # until restart.
        store.scheduler_loop = asyncio.get_running_loop()
        brief_on = getattr(config, "daily_brief", None) and config.daily_brief.enabled
        diary_on = getattr(config, "diary", None) and config.diary.enabled
        reachout_on = getattr(config, "reach_out", None) and config.reach_out.enabled
        followthrough_on = (
            getattr(config, "follow_through", None) and config.follow_through.enabled
        )
        life_on = getattr(config, "life_threads", None) and config.life_threads.enabled
        insights_on = getattr(config, "insights", None) and config.insights.enabled
        self_eval_on = getattr(config, "self_eval", None) and config.self_eval.enabled
        async_notify_on = (
            getattr(config, "async_notify", None) and config.async_notify.enabled
        )
        if (
            brief_on
            or diary_on
            or reachout_on
            or followthrough_on
            or life_on
            or insights_on
            or self_eval_on
            or async_notify_on
        ):
            from .scheduler import BackgroundScheduler

            scheduler = BackgroundScheduler(store, config)
            await scheduler.start()
        app.state.scheduler = scheduler
        # The store carries the live scheduler handle (None when the gate
        # didn't fire): settings_api's reload path checks it before starting
        # one, and shutdown below stops whichever instance ended up running —
        # boot-started or settings-started.
        store.scheduler = scheduler
        yield
        running = getattr(store, "scheduler", None)
        if running is not None:
            await running.stop()

    app = FastAPI(title="Amadeus — Kurisu Web Lab", lifespan=lifespan)
    app.state.session_store = store  # exposed for introspection/tests
    # Wildcard origins with credentials is rejected by the CORS spec; the API
    # identifies users via request-body IDs, not cookies, so credentials stay off.
    cors_origins = list(getattr(config.web, "cors_origins", None) or ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Public-demo hardening (web.demo_mode / web.rate_limit_per_minute).
    # Demo mode locks the settings/persona mutation surface: a hosted demo's
    # visitors must not be able to rewrite providers, keys, or the persona
    # identity. The limiter caps the LLM/TTS/STT-invoking endpoints per
    # client IP; the WS chat path checks app.state.turn_limiter itself
    # (WebSockets don't traverse HTTP middleware).
    demo_mode = bool(getattr(config.web, "demo_mode", False))
    rate_limit = getattr(config.web, "rate_limit_per_minute", None)
    if demo_mode and not rate_limit:
        rate_limit = 10
    limiter = TurnRateLimiter(rate_limit) if rate_limit else None
    app.state.turn_limiter = limiter

    if demo_mode:

        @app.middleware("http")
        async def demo_readonly_guard(request: Request, call_next):
            path = request.url.path
            if request.method != "GET" and (
                path.startswith("/api/settings")
                or path.startswith("/api/persona")
                or path.startswith("/api/characters")
                or path == "/api/memory/import"
            ):
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "This is a read-only public demo — settings are locked."
                    },
                )
            return await call_next(request)

    if limiter is not None:

        @app.middleware("http")
        async def turn_rate_limit(request: Request, call_next):
            if request.method == "POST" and request.url.path.startswith(
                ("/api/chat", "/api/tts", "/api/stt", "/api/screen_peek")
            ):
                key = client_key(
                    request.headers,
                    request.client.host if request.client else None,
                )
                if not limiter.allow(key):
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Rate limit reached — give it a minute, this instance is shared."
                        },
                    )
            return await call_next(request)

    # Optional bearer-token auth. When web.api_token (or AMADEUS_API_TOKEN)
    # is unset the middleware passes everything through untouched — zero
    # friction for local use. When set, every /api/* route requires
    # "Authorization: Bearer <token>"; /healthz, docs, and static/SPA routes
    # stay open. The WS endpoint checks ?token=... itself (WebSockets don't
    # traverse HTTP middleware, and browsers can't set WS headers).
    api_token = getattr(config.web, "api_token", None) or None

    def _token_ok(candidate: Optional[str]) -> bool:
        if api_token is None:
            return True
        return bool(candidate) and secrets.compare_digest(candidate, api_token)

    if api_token is not None:

        @app.middleware("http")
        async def require_bearer_token(request: Request, call_next):
            if request.url.path.startswith("/api/"):
                header = request.headers.get("authorization", "")
                scheme, _, credentials = header.partition(" ")
                if scheme.lower() != "bearer" or not _token_ok(credentials.strip()):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Missing or invalid bearer token."},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            return await call_next(request)

    # Mount static files directory. Packaged desktop builds relocate assets
    # into Tauri resources, while source/dev runs keep using the repo root.
    static_dir_override = os.getenv("AMADEUS_STATIC_DIR")
    static_dir = (
        Path(static_dir_override)
        if static_dir_override
        else Path(__file__).resolve().parent.parent.parent / "static"
    )
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    frontend_dir_override = os.getenv("KURISU_FRONTEND_DIR")
    default_frontend_dir = (
        Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    )
    frontend_dir = (
        Path(frontend_dir_override) if frontend_dir_override else default_frontend_dir
    )
    use_frontend_bundle = (
        frontend_dir.is_dir() and (frontend_dir / "index.html").exists()
    )

    if use_frontend_bundle:
        frontend_index_html = (frontend_dir / "index.html").read_text(encoding="utf-8")
        assets_dir = frontend_dir / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="frontend-assets",
            )

        vite_svg_path = frontend_dir / "vite.svg"
        if vite_svg_path.exists():

            @app.get("/vite.svg")
            async def vite_svg() -> FileResponse:
                return FileResponse(vite_svg_path, media_type="image/svg+xml")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(frontend_index_html)

        @app.get("/mobile", response_class=HTMLResponse)
        async def mobile() -> HTMLResponse:
            return HTMLResponse(frontend_index_html)

    else:
        # The React app in frontend/ is the only client. Serve a minimal
        # pointer page instead of a second inline implementation.
        not_built_html = (
            '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            '<title>Kurisu</title></head><body style="font-family: sans-serif; '
            'max-width: 40rem; margin: 4rem auto;">'
            "<h1>Kurisu — frontend not built</h1>"
            "<p>The web client bundle was not found. Build it with:</p>"
            "<pre>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>"
            "<p>then restart the server (or set <code>KURISU_FRONTEND_DIR</code> "
            "to a built bundle). The API itself is running — see "
            '<a href="/healthz">/healthz</a>.</p></body></html>'
        )

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(not_built_html)

        @app.get("/mobile", response_class=HTMLResponse)
        async def mobile() -> HTMLResponse:
            return HTMLResponse(not_built_html)

    # API routes, split by concern. Registration order mirrors the original
    # single-file layout; the SPA wildcard below must still come last.
    app.include_router(routes_meta.build_router(store, config))
    app.include_router(
        routes_chat.build_router(store, config, api_token=api_token, token_ok=_token_ok)
    )
    app.include_router(routes_sessions.build_router(store))
    app.include_router(routes_persona.build_router(store))
    app.include_router(routes_characters.build_router(store))

    # Wildcard fallback route registered last so it doesn't intercept specific routes
    if use_frontend_bundle:

        @app.get(
            "/{full_path:path}", response_class=HTMLResponse, include_in_schema=False
        )
        async def spa_fallback(full_path: str) -> HTMLResponse:
            blocked_prefixes = ("api", "static", "assets")
            blocked_exact = {"healthz", "docs", "redoc", "openapi.json"}
            if (
                not full_path
                or full_path in blocked_exact
                or full_path.startswith(blocked_prefixes)
                or "." in full_path
            ):
                raise HTTPException(status_code=404)
            return HTMLResponse(frontend_index_html)

    return app


def _module_app() -> Optional[FastAPI]:
    # Serving entrypoints (uvicorn amadeus.web.app:app, Docker, desktop
    # shells) need an eager module-level app. Tests build their own via
    # create_app(); skip the eager one there, because it loads the real
    # providers from config.local.yaml — which can mean gigabytes of TTS
    # model weights pulled in at import/collection time.
    import sys

    if "pytest" in sys.modules or os.getenv("AMADEUS_NO_MODULE_APP"):
        return None
    return create_app()


app = _module_app()
