from pathlib import Path

import pytest

from teametrics.common.async_save import submit_copy, temporary_path, wait_for_pending_copies


def test_background_copy_is_atomic_and_cleans_source(tmp_path):
    source = temporary_path(tmp_path / "result.nc")
    destination = tmp_path / "nested" / "result.nc"
    source.write_bytes(b"result")

    submit_copy(source, destination)
    wait_for_pending_copies()

    assert destination.read_bytes() == b"result"
    assert not source.exists()


def test_background_copy_propagates_errors(tmp_path):
    source = tmp_path / "missing.nc"
    destination = tmp_path / "result.nc"

    submit_copy(source, destination)
    with pytest.raises(FileNotFoundError):
        wait_for_pending_copies()
