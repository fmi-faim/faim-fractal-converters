from ome_zarr_converters_tools import (
    ConverterOptions,
    OmeZarrOptions,
    PerFovGrouping,
    exec_compound_task,
)
from pathlib import Path
import pytest

from faim_fractal_converters import convert_visiview_init_task
from faim_fractal_converters import single_image_compute_task


@pytest.fixture
def companion_ome_file():
    return (
        Path(__file__).parent.parent
        / "data"
        / "20260624_4pos_2simCh_3planes_5timepoints"
        / "20260624_4pos_2simch_3planes_5time_1.companion.ome"
    )


def test_compound_converter_task(companion_ome_file, tmpdir):
    """Test the compound converter task."""
    init_task_kwargs = {
        "zarr_dir": str(tmpdir / "test_visiview.zarr"),
        "companion_ome_file": companion_ome_file.as_posix(),
        "converter_options": ConverterOptions(
            omezarr_options=OmeZarrOptions(
                ngff_version="0.5",
                table_backend="csv",
            ),
            grouping=PerFovGrouping(),
        ),
    }
    updates = exec_compound_task(
        init_task_fn=convert_visiview_init_task,
        compute_task_fn=single_image_compute_task,
        init_task_kwargs=init_task_kwargs,
    )
    assert len(updates) == 4
