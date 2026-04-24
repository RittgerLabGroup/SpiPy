from pathlib import Path

from spires.logging_utils import configure_spires_file_logger, format_log_event, log_event


def test_format_log_event_is_plain_text_and_structured():
    message = format_log_event(
        "open_viirs_surface_reflectance",
        input_path="/tmp/example.h5",
        selected_bands=["I1", "M4"],
    )

    assert 'event="open_viirs_surface_reflectance"' in message
    assert 'input_path="/tmp/example.h5"' in message
    assert 'selected_bands=["I1", "M4"]' in message


def test_configure_spires_file_logger_writes_log_file(tmp_path):
    log_path = tmp_path / "spires_reader.log"
    logger = configure_spires_file_logger(log_path, logger_name="spires.test", log_to_stdout=False)

    log_event(
        logger,
        "open_viirs_surface_reflectance",
        input_path=Path("/tmp/example.h5"),
        lut_file="/tmp/lut_viirs.mat",
        selected_bands=["I1", "I2", "M4"],
    )
    for handler in logger.handlers:
        handler.flush()

    contents = log_path.read_text()
    assert "INFO spires.test" in contents
    assert 'event="open_viirs_surface_reflectance"' in contents
    assert 'input_path="/tmp/example.h5"' in contents
    assert 'lut_file="/tmp/lut_viirs.mat"' in contents
