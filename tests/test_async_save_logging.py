from teametrics.common.async_save import submit_copy, temporary_path, wait_for_pending_copies


def test_async_copy_logs_temporary_and_destination_paths(tmp_path, caplog):
    source = temporary_path(tmp_path / "result.nc")
    destination = tmp_path / "output" / "result.nc"
    source.write_bytes(b"result")

    with caplog.at_level("INFO"):
        submit_copy(source, destination)
        wait_for_pending_copies()

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert str(source) in messages
    assert str(destination) in messages
