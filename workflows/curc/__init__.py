"""CURC-specific workflow orchestration for large-scale SpiPy runs."""

from workflows.curc.config import CurcWorkflowConfig, SlurmProfile
from workflows.curc.manifest import PlannedJob, build_job_manifest
from workflows.curc.planner import plan_viirs_snpp_inversion_array, plan_viirs_snpp_workflow_steps
from workflows.curc.runtime import (
    InversionRuntimeContext,
    build_viirs_snpp_inversion_runtime_context,
    execute_viirs_snpp_inversion_task,
)
from workflows.curc.status import (
    InversionArrayStatusReport,
    InversionTaskStatus,
    scan_inversion_array_status,
    should_auto_retry,
    write_retry_manifest,
)
from workflows.curc.slurm import (
    render_array_submission_payload_from_manifest,
    render_sbatch_command_for_array_payload,
)
from workflows.curc.steps import InversionTaskPlan, SlurmArrayPlan, WorkflowStepPlan
from workflows.curc.task_manifest import (
    resolve_inversion_task_from_manifest,
    write_inversion_array_manifest,
)

__all__ = [
    "CurcWorkflowConfig",
    "InversionArrayStatusReport",
    "InversionTaskPlan",
    "InversionRuntimeContext",
    "InversionTaskStatus",
    "PlannedJob",
    "SlurmArrayPlan",
    "SlurmProfile",
    "WorkflowStepPlan",
    "build_job_manifest",
    "build_viirs_snpp_inversion_runtime_context",
    "execute_viirs_snpp_inversion_task",
    "plan_viirs_snpp_inversion_array",
    "plan_viirs_snpp_workflow_steps",
    "resolve_inversion_task_from_manifest",
    "scan_inversion_array_status",
    "should_auto_retry",
    "render_array_submission_payload_from_manifest",
    "render_sbatch_command_for_array_payload",
    "write_inversion_array_manifest",
    "write_retry_manifest",
]
