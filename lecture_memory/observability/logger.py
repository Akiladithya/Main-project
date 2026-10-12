"""
observability/logger.py
-----------------------
Structured logging foundation. Every pipeline stage should use this
instead of bare print() or ad-hoc logging calls.

Records: stage name, duration, model used, scores, failures.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Optional


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


@contextmanager
def stage_timer(logger: logging.Logger, stage: str, **context: Any):
    """Context manager that logs stage entry, exit, and elapsed time."""
    logger.info(f"stage_start | stage={stage} | context={context}")
    t0 = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - t0
        logger.info(f"stage_end | stage={stage} | elapsed={elapsed:.3f}s | status=ok")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error(
            f"stage_end | stage={stage} | elapsed={elapsed:.3f}s | status=failed | error={exc}"
        )
        raise


def log_retrieval(
    logger: logging.Logger,
    query: str,
    n_candidates: int,
    top_score: Optional[float],
) -> None:
    logger.info(
        f"retrieval | query_len={len(query)} | candidates={n_candidates} | top_score={top_score}"
    )


def log_verification(
    logger: logging.Logger,
    n_claims: int,
    n_supported: int,
    n_unsupported: int,
) -> None:
    logger.info(
        f"verification | claims={n_claims} | supported={n_supported} | unsupported={n_unsupported}"
    )
