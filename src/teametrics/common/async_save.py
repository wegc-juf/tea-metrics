"""Background file-copy support for completed result files."""

import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from .TEA_logger import logger


_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tea-save")
_pending = []
_lock = Lock()


def submit_copy(source, destination):
    """Copy a completed temporary file in the background."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Queueing async copy from temporary file {source} to {destination}")
    future = _executor.submit(_copy_atomically, source, destination)
    with _lock:
        _pending.append(future)


def wait_for_pending_copies():
    """Wait for all queued copies and propagate copy errors."""
    logger.debug("Waiting for pending async file copies")
    while True:
        with _lock:
            if not _pending:
                logger.debug("No pending async file copies")
                return
            futures = _pending[:]
            _pending.clear()
        for future in futures:
            future.result()


def temporary_path(destination):
    """Create a temporary path for output that will later be copied."""
    suffix = Path(destination).suffix
    handle = tempfile.NamedTemporaryFile(prefix="tea-", suffix=suffix, delete=False)
    path = Path(handle.name)
    handle.close()
    path.unlink()
    return path


def _copy_atomically(source, destination):
    destination_tmp = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, destination_tmp)
        destination_tmp.replace(destination)
        logger.info(f"Completed async copy to {destination}")
    finally:
        source.unlink(missing_ok=True)
        destination_tmp.unlink(missing_ok=True)
