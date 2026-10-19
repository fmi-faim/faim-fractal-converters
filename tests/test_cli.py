from pathlib import Path

import pytest

from faim_fractal_converters import cli
from faim_fractal_converters.convert_visiview_init_task import (
    convert_visiview_init_task,
)
from faim_fractal_converters.convert_visiview_nd_init_task import (
    convert_visiview_nd_init_task,
)
from faim_fractal_converters.single_image_compute_task import single_image_compute_task


def _run_cli(arguments: list[str]) -> None:
    command, bound_arguments, _ = cli.app.parse_args(arguments)
    command(*bound_arguments.args, **bound_arguments.kwargs)


def test_companion_command_accepts_repeated_files(tmp_path, monkeypatch):
    first_file = tmp_path / "first.companion.ome"
    second_file = tmp_path / "second.companion.ome"
    first_file.touch()
    second_file.touch()
    zarr_dir = tmp_path / "output"
    calls = []

    def fake_exec_compound_task(**kwargs):
        calls.append(kwargs)
        return [{}, {}]

    monkeypatch.setattr(cli, "exec_compound_task", fake_exec_compound_task)

    _run_cli(
        [
            "companion",
            "--file",
            str(first_file),
            "--file",
            str(second_file),
            "--zarr-dir",
            str(zarr_dir),
        ]
    )

    assert calls == [
        {
            "init_task_fn": convert_visiview_init_task,
            "compute_task_fn": single_image_compute_task,
            "init_task_kwargs": {
                "zarr_dir": zarr_dir.as_posix(),
                "input_options": {
                    "mode": "list",
                    "ome_file_list": [first_file.as_posix(), second_file.as_posix()],
                },
            },
        }
    ]


def test_nd_command_uses_non_recursive_folder_input(tmp_path, monkeypatch):
    first_file = tmp_path / "first.nd"
    second_file = tmp_path / "second.nd"
    nested_file = tmp_path / "nested" / "ignored.nd"
    first_file.touch()
    second_file.touch()
    nested_file.parent.mkdir()
    nested_file.touch()
    zarr_dir = tmp_path / "output"
    calls = []

    def fake_exec_compound_task(**kwargs):
        calls.append(kwargs)
        return [{}]

    monkeypatch.setattr(cli, "exec_compound_task", fake_exec_compound_task)

    _run_cli(["nd", "--folder", str(tmp_path), "--zarr-dir", str(zarr_dir)])

    assert calls == [
        {
            "init_task_fn": convert_visiview_nd_init_task,
            "compute_task_fn": single_image_compute_task,
            "init_task_kwargs": {
                "zarr_dir": zarr_dir.as_posix(),
                "input_options": {
                    "mode": "list",
                    "nd_file_list": [first_file.as_posix(), second_file.as_posix()],
                },
            },
        }
    ]


@pytest.mark.parametrize(
    ("file", "folder"),
    [
        (None, None),
        ([Path("input.companion.ome")], Path("input-folder")),
    ],
)
def test_companion_requires_exactly_one_input_mode(file, folder, tmp_path):
    with pytest.raises(
        ValueError, match="either one or more --file options or --folder"
    ):
        cli.companion(file=file, folder=folder, zarr_dir=tmp_path / "output")


def test_companion_rejects_wrong_file_type(tmp_path):
    input_file = tmp_path / "input.nd"
    input_file.touch()

    with pytest.raises(ValueError, match="Expected .companion.ome input file"):
        cli.companion(
            file=[input_file],
            folder=None,
            zarr_dir=tmp_path / "output",
        )


def test_nd_folder_without_matching_files_fails(tmp_path):
    with pytest.raises(cli.CLIInputError, match="No .nd files found"):
        cli.nd(file=None, folder=tmp_path, zarr_dir=tmp_path / "output")


def test_main_exits_with_cli_error_status(monkeypatch):
    def raise_input_error():
        raise cli.CLIInputError("Invalid CLI input")

    monkeypatch.setattr(cli, "app", raise_input_error)

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
