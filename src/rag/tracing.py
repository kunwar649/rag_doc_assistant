"""Optional Langfuse tracing that can never break the app.

If the langfuse package isn't installed, keys aren't set, or the SDK raises
(network down, API change), every call silently becomes a no-op. Tracing is a
diagnostic aid, so it must not take answers down with it.

Observations are started and ended explicitly instead of with context managers:
FastAPI runs streaming generators across worker threads, and context-manager
spans rely on thread-local "current span" state that doesn't survive that.

Trace shape per question:
    rag-query (span)            question in, answer out
    ├── rewrite (generation)    only for follow-ups
    ├── retrieve (span)         query in, cited pages + scores out
    └── generate (generation)   prompt in, answer out, model + token usage
"""
import logging
import os

from .config import Settings

logger = logging.getLogger(__name__)


class _NoopObservation:
    def child(self, *args, **kwargs) -> "_NoopObservation":
        return self

    def update(self, **kwargs) -> None:
        pass

    def end(self, **kwargs) -> None:
        pass


NOOP = _NoopObservation()


class _Observation:
    """Wraps a Langfuse observation; disables itself after the first SDK error."""

    def __init__(self, obs):
        self._obs = obs

    def child(self, name: str, as_type: str = "span", **kwargs):
        try:
            return _Observation(self._obs.start_observation(name=name, as_type=as_type, **kwargs))
        except Exception:
            logger.debug("Langfuse child observation failed", exc_info=True)
            return NOOP

    def update(self, **kwargs) -> None:
        try:
            self._obs.update(**kwargs)
        except Exception:
            logger.debug("Langfuse update failed", exc_info=True)

    def end(self, **kwargs) -> None:
        try:
            if kwargs:
                self._obs.update(**kwargs)
            self._obs.end()
        except Exception:
            logger.debug("Langfuse end failed", exc_info=True)


class Tracer:
    def __init__(self, client=None):
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def start(self, name: str, **kwargs):
        if not self._client:
            return NOOP
        try:
            return _Observation(self._client.start_observation(name=name, as_type="span", **kwargs))
        except Exception:
            logger.warning("Langfuse tracing failed; continuing without it", exc_info=True)
            return NOOP

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception:
                logger.debug("Langfuse flush failed", exc_info=True)


def create_tracer(s: Settings) -> Tracer:
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return Tracer()
    try:
        from langfuse import get_client
    except ImportError:
        logger.warning("Langfuse keys are set but the langfuse package isn't installed; tracing disabled")
        return Tracer()
    # The SDK reads its settings from environment variables. Both host variable
    # names are set because SDK versions differ on which one they read.
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", s.langfuse_host)
    os.environ.setdefault("LANGFUSE_BASE_URL", s.langfuse_host)
    try:
        return Tracer(get_client())
    except Exception:
        logger.warning("Could not start Langfuse; tracing disabled", exc_info=True)
        return Tracer()
