"""Errors shown to Marco in the notebooks: the message says in plain English what to do next."""

from __future__ import annotations


class StageIncomplete(Exception):
    """A paid or manual step did not finish; the message tells Marco what to do next."""
