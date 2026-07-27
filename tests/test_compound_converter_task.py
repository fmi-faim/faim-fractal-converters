from ome_zarr_converters_tools import (
    ConverterOptions,
    OmeZarrOptions,
    PerFovGrouping,
    exec_compound_task,
)
from ngio import open_image
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
        "input_options": dict(
            mode="list",
            ome_file_list=[str(companion_ome_file)],
        ),
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

    representative_update = updates[0]
    if "zarr_url" in representative_update:
        output_zarr_url = representative_update["zarr_url"]
    elif (
        "image_list_updates" in representative_update
        and representative_update["image_list_updates"]
    ):
        output_zarr_url = representative_update["image_list_updates"][0]["zarr_url"]
    else:
        pytest.fail("Could not find output zarr URL in compound task updates")

    output_image = open_image(output_zarr_url)
    assert output_image.shape == (5, 2, 3, 512, 512)
