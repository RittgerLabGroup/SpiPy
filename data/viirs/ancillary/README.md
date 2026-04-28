# VIIRS Ancillary Inputs

This directory mirrors the intended batch-processing layout for reusable VIIRS ancillary inputs.

```text
viirs/
  ancillary/
    global/
    tiles/
      h08v05/
        static/
        annual/
        logs/
```

Use `spires.sensors.viirs.viirs_ancillary_path(...)` and related helpers to build paths from a configurable data root.

For local development, the data root is `data`, so tile-level ancillary paths resolve under `data/viirs/ancillary/tiles/...`.
