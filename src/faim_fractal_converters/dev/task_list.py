"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ConverterCompoundTask,
)

AUTHORS = "Facility for Advanced Imaging and Microscopy (FAIM), FMI, Basel"

DOCS_LINK = None


TASK_LIST = [
    ConverterCompoundTask(
        name="Convert Visiview companion.ome",
        executable_init="convert_visiview_init_task.py",
        executable="single_image_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 12000},
        meta={"cpus_per_task": 1, "mem": 12000},
        category="Conversion",
        tags=[
            "Visiview",
        ],
    )
]
