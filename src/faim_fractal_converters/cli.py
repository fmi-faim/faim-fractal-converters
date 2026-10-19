from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter
from loguru import logger
from ome_zarr_converters_tools import exec_compound_task

from faim_fractal_converters.convert_visiview_init_task import (
    convert_visiview_init_task,
)
from faim_fractal_converters.convert_visiview_nd_init_task import (
    convert_visiview_nd_init_task,
)
from faim_fractal_converters.single_image_compute_task import single_image_compute_task

app = App(name="convert", help="Convert Visiview acquisitions to OME-Zarr.")


class CLIInputError(ValueError):
    """Invalid input to a conversion command."""


def _convert(
    *,
    input_format: Literal["companion", "nd"],
    files: list[Path] | None,
    folder: Path | None,
    zarr_dir: Path,
) -> None:
    if (files is None) == (folder is None):
        raise CLIInputError("Provide either one or more --file options or --folder.")

    file_pattern, expected_suffix = (
        ("*.companion.ome", ".companion.ome")
        if input_format == "companion"
        else ("*.nd", ".nd")
    )

    if folder is not None:
        folder = folder.expanduser()
        if not folder.exists():
            raise CLIInputError(f"Input folder not found: {folder}")
        if not folder.is_dir():
            raise CLIInputError(f"Input path is not a folder: {folder}")
        input_files = sorted(folder.glob(file_pattern))
        if not input_files:
            raise CLIInputError(f"No {expected_suffix} files found in folder: {folder}")
    else:
        input_files = [path.expanduser() for path in files or []]

    for input_file in input_files:
        if not input_file.exists():
            raise CLIInputError(f"Input file not found: {input_file}")
        if not input_file.is_file():
            raise CLIInputError(f"Input path is not a file: {input_file}")
        if not input_file.name.endswith(expected_suffix):
            raise CLIInputError(
                f"Expected {expected_suffix} input file, got: {input_file}"
            )

    if input_format == "companion":
        init_task_fn = convert_visiview_init_task
        input_options = {
            "mode": "list",
            "ome_file_list": [path.as_posix() for path in input_files],
        }
    else:
        init_task_fn = convert_visiview_nd_init_task
        input_options = {
            "mode": "list",
            "nd_file_list": [path.as_posix() for path in input_files],
        }

    updates = exec_compound_task(
        init_task_fn=init_task_fn,
        compute_task_fn=single_image_compute_task,
        init_task_kwargs={
            "zarr_dir": zarr_dir.expanduser().as_posix(),
            "input_options": input_options,
        },
    )
    logger.info(
        f"Converted {len(input_files)} {input_format} input file(s) into "
        f"{len(updates)} OME-Zarr dataset(s) in {zarr_dir}."
    )


@app.command(name="companion")
def companion(
    *,
    file: Annotated[
        list[Path] | None,
        Parameter(
            name="file",
            allow_repeating=True,
            help="Companion OME file to convert. Repeat to provide multiple files.",
        ),
    ] = None,
    folder: Path | None = None,
    zarr_dir: Path,
) -> None:
    """Convert one or more Visiview *.companion.ome files."""
    _convert(
        input_format="companion",
        files=file,
        folder=folder,
        zarr_dir=zarr_dir,
    )


@app.command(name="nd")
def nd(
    *,
    file: Annotated[
        list[Path] | None,
        Parameter(
            name="file",
            allow_repeating=True,
            help="Visiview ND file to convert. Repeat to provide multiple files.",
        ),
    ] = None,
    folder: Path | None = None,
    zarr_dir: Path,
) -> None:
    """Convert one or more Visiview *.nd files."""
    _convert(
        input_format="nd",
        files=file,
        folder=folder,
        zarr_dir=zarr_dir,
    )


def main() -> None:
    try:
        app()
    except CLIInputError as error:
        logger.error(str(error))
        raise SystemExit(2) from None
