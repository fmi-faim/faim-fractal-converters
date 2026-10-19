from pathlib import Path

import pytest
from ome_zarr_converters_tools import ConverterOptions, PerFovGrouping

from faim_fractal_converters.convert_visiview_nd_init_task import (
    convert_visiview_nd_init_task,
)


@pytest.fixture
def nd_file():
    return (
        Path(__file__).parent.parent
        / "data"
        / "benchmark_v1_2018_x64y64z5c2s4t11"
        / "benchmark_v1_2018_x64y64z5c2s4t11.nd"
    )


def test_convert_visiview_nd_init_task(nd_file):
    parallelization_list = convert_visiview_nd_init_task(
        zarr_dir="/tmp/test_visiview_nd.zarr",
        input_options={
            "mode": "list",
            "nd_file_list": [str(nd_file)],
        },
        converter_options=ConverterOptions(
            grouping=PerFovGrouping(),
        ),
    )

    assert parallelization_list is not None
    assert len(parallelization_list["parallelization_list"]) == 4
