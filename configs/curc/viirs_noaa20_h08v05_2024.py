"""Example CURC workflow config for VIIRS NOAA-20 tile h08v05 in 2024."""

from pathlib import Path

from workflows.curc.config import CurcWorkflowConfig


CONFIG = CurcWorkflowConfig(
    scratch_root=Path("/scratch/alpine/ropa5718/spipy"),
    input_source_root=Path("/pl/active/rittgerlab/INPUTS_TO_BE_DEFINED"),
    sensor="viirs",
    platforms=("noaa20",),
    tiles=("h08v05",),
    years=(2024,),
    date_glob="*",
    dry_run=True,
    apply_valid_inversion_mask=False,
    use_grouping=True,
    grouping_method="chunk_bin_mean",
)
