from pathlib import Path

from spires.logging_utils import configure_spires_file_logger, format_log_event, log_event, make_spires_log_path


def test_format_log_event_is_plain_text_and_structured():
    message = format_log_event(
        "open_viirs_surface_reflectance",
        stage="reader",
        event_type="detail",
        input_path="/tmp/example.h5",
        selected_bands=["I1", "M4"],
    )

    assert 'event="open_viirs_surface_reflectance"' in message
    assert message.index('stage="reader"') < message.index('input_path="/tmp/example.h5"')
    assert 'input_path="/tmp/example.h5"' in message
    assert 'selected_bands="I1,M4"' in message


def test_format_log_event_adds_visual_prefix_for_summary_messages():
    message = format_log_event(
        "build_viirs_timeseries",
        stage="timeseries",
        event_type="summary",
        status="completed",
    )

    assert message.startswith('====== SUMMARY ====== event="build_viirs_timeseries"')


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


def test_configure_spires_file_logger_adds_separator_line_before_summary(tmp_path):
    log_path = tmp_path / "spires_summary.log"
    logger = configure_spires_file_logger(log_path, logger_name="spires.test", log_to_stdout=False)

    log_event(
        logger,
        "build_viirs_timeseries",
        stage="timeseries",
        event_type="summary",
        status="completed",
    )
    for handler in logger.handlers:
        handler.flush()

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert set(lines[0]) == {"="}
    expected_prefix = "INFO spires.test ====== SUMMARY ====== event=\"build_viirs_timeseries\""
    assert expected_prefix in lines[1]
    prefix_without_message = lines[1].split(" ====== SUMMARY ======")[0]
    assert len(lines[0]) == len(prefix_without_message)


def test_configure_spires_file_logger_applies_context_indentation(tmp_path):
    log_path = tmp_path / "spires_indent.log"
    logger = configure_spires_file_logger(log_path, logger_name="spires.test.indent", log_to_stdout=False)

    log_event(
        logger,
        "curc_run_viirs_snpp_inversion_task",
        stage="curc_runtime",
        event_type="start",
        status="started",
    )
    log_event(
        logger,
        "open_viirs_surface_reflectance",
        stage="reader",
        event_type="detail",
        status="started",
    )
    log_event(
        logger,
        "curc_run_viirs_snpp_inversion_task",
        stage="curc_runtime",
        event_type="summary",
        status="completed",
    )
    for handler in logger.handlers:
        handler.flush()

    lines = log_path.read_text().splitlines()
    event_lines = [line for line in lines if 'event="' in line]
    assert any('event="curc_run_viirs_snpp_inversion_task"' in line and "    event=" not in line for line in event_lines)
    assert any('event="open_viirs_surface_reflectance"' in line and "    event=" in line for line in event_lines)


def test_configure_spires_file_logger_writes_to_aggregate_sink(tmp_path):
    log_path = tmp_path / "spires_primary.log"
    aggregate_path = tmp_path / "spires_aggregate.log"
    logger = configure_spires_file_logger(
        log_path,
        logger_name="spires.test.aggregate",
        log_to_stdout=False,
        aggregate_log_path=aggregate_path,
    )
    log_event(logger, "open_viirs_surface_reflectance", event_type="detail", stage="reader")
    for handler in logger.handlers:
        handler.flush()

    primary_contents = log_path.read_text()
    aggregate_contents = aggregate_path.read_text()
    assert 'event="open_viirs_surface_reflectance"' in primary_contents
    assert 'event="open_viirs_surface_reflectance"' in aggregate_contents


def test_make_spires_log_path_creates_timestamped_log_name(tmp_path):
    log_path = make_spires_log_path(
        tmp_path,
        prefix="viirs_r0",
        tile="h08v05",
        sensor="snpp",
        timestamp="20260424_194828",
    )

    assert log_path.parent == tmp_path.resolve()
    assert log_path.name == "viirs_r0_snpp_h08v05_20260424_194828.log"
