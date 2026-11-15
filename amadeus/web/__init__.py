"""Web application components for the Amadeus assistant."""

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
