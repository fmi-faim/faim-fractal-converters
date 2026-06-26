import pytest

from faim_fractal_converters.convert_visiview_init_task import (
    convert_visiview_init_task,
)
from pathlib import Path


@pytest.fixture
def companion_ome_file():
    return (
        Path(__file__).parent.parent
        / "data"
        / "20260624_4pos_2simCh_3planes_5timepoints"
        / "20260624_4pos_2simch_3planes_5time_1.companion.ome"
    )


def test_convert_visiview_init_task(companion_ome_file):
    parallelization_list = convert_visiview_init_task(
        zarr_dir="/tmp/test_visiview.zarr",
        companion_ome_file=str(companion_ome_file),
        sanitize_metadata=True,
    )
    assert parallelization_list is not None
    number_of_images = 4
    assert len(parallelization_list["parallelization_list"]) == number_of_images
