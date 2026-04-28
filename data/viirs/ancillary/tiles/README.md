# Tile Ancillary Products

Tile directories should be named with the VIIRS sinusoidal tile id, for example `h08v05`.

Each tile may contain:

- `static/`: products reused across years
- `annual/<year>/`: year-specific products such as R0
- `logs/`: local processing logs for ancillary generation
