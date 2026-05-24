"""CURC-specific workflow orchestration for large-scale SpiPy runs."""

from workflows.curc.config import CurcWorkflowConfig, SlurmProfile
from workflows.curc.execution import (
    execute_viirs_snpp_workflow_step,
    preview_viirs_snpp_workflow_step_execution,
    resolve_viirs_snpp_workflow_step,
)
from workflows.curc.manifest import PlannedJob, build_job_manifest
from workflows.curc.planner import plan_viirs_snpp_inversion_array, plan_viirs_snpp_workflow_steps
from workflows.curc.runner import preview_viirs_snpp_step_execution, run_viirs_snpp_step
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
    write_status_summary_artifacts,
    write_retry_manifest,
)
from workflows.curc.slurm import (
    render_array_submission_payload_from_manifest,
    render_sbatch_command_for_array_payload,
    render_sbatch_command_for_finalize_wrap,
)
from workflows.curc.steps import InversionTaskPlan, SlurmArrayPlan, WorkflowStepPlan
from workflows.curc.task_manifest import (
    resolve_inversion_task_from_manifest,
    write_inversion_array_manifest,
)

__all__ = [
    "CurcWorkflowConfig",
    "execute_viirs_snpp_workflow_step",
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
    "preview_viirs_snpp_step_execution",
    "preview_viirs_snpp_workflow_step_execution",
    "resolve_inversion_task_from_manifest",
    "resolve_viirs_snpp_workflow_step",
    "run_viirs_snpp_step",
    "scan_inversion_array_status",
    "should_auto_retry",
    "render_array_submission_payload_from_manifest",
    "render_sbatch_command_for_array_payload",
    "render_sbatch_command_for_finalize_wrap",
    "write_inversion_array_manifest",
    "write_status_summary_artifacts",
    "write_retry_manifest",
]
