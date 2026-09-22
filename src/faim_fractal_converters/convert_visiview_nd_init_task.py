from pathlib import Path
from typing import Annotated, Any, Literal

from loguru import logger
from metamorph_mda_parser.nd import ImageFile, NdInfo
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    ChannelInfo,
    ConverterOptions,
    DefaultImageLoader,
    ImplementedFilters,
    OverwriteMode,
    SingleImage,
    Tile,
    setup_images_for_conversion,
    tiles_aggregation_pipeline,
)
from pydantic import BaseModel, Field, validate_call
from tifffile import TiffFile

DefaultConverterOptions = ConverterOptions()


class FolderLoading(BaseModel):
    """Load companion.ome files from folder."""

    mode: Literal["folder"] = "folder"
    folder_path: str
    glob_pattern: str = "*.nd"


class ListLoading(BaseModel):
    """Load a list of specified Visiview .nd files."""

    mode: Literal["list"] = "list"
    nd_file_list: list[str]


Loading = Annotated[
    FolderLoading | ListLoading,
    Field(discriminator="mode"),
]


def _generate_channel_info_list(nd_info):
    return [
        ChannelInfo(
            channel_label=wave_name,
            wavelength_id=None,
            color=None,
        )
        for wave_name in nd_info.wave_names
    ]


def _get_xy_pixel_size(tile: ImageFile) -> float:
    with TiffFile(tile.path) as stk:
        return stk.stk_metadata["XCalibration"]


def tile_list_from_single_nd(
    nd_path: Path,
) -> list[Tile]:
    nd_info = NdInfo.from_path(nd_path, fix_decimal_comma=True)
    tile_files = nd_info.get_tiles()
    tiles: list[Tile] = []
    logger.info(f"The file ({nd_path.name}) defines {len(tile_files)} images...")

    collection = SingleImage(image_path=nd_path.stem)
    xy_pixel_size = _get_xy_pixel_size(tile_files[0])
    acquisition_details = AcquisitionDetails(
        start_t_space="pixel",
        xy_pixel_size=xy_pixel_size or None,
        z_spacing=nd_info.z_step_size,
        channels=_generate_channel_info_list(nd_info),
    )

    for tile_file in tile_files:
        with TiffFile(tile_file.path) as stk:
            shape = stk.pages[0].shape
            stage_position = stk.stk_metadata["StagePosition"]
        tiles.append(
            Tile(
                fov_name=f"s{tile_file.position}",
                start_x=float(stage_position[0][0]),
                start_y=float(stage_position[0][1]),
                start_z=0,
                start_c=tile_file.channel,
                start_t=tile_file.time,
                length_x=shape[1],
                length_y=shape[0],
                length_z=nd_info.n_z_steps,
                length_c=1,
                length_t=1,
                attributes={},
                collection=collection,
                image_loader=DefaultImageLoader(
                    file_path=tile_file.path.as_posix(),
                ),
                acquisition_details=acquisition_details,
            )
        )

    return tiles


@validate_call
def convert_visiview_nd_init_task(
    *,
    zarr_dir: str,
    input_options: Loading,
    converter_options: ConverterOptions = DefaultConverterOptions,
    filters: list[ImplementedFilters] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Initialization task for Visiview .nd files."""
    converter_options = ConverterOptions.model_validate(converter_options)

    if input_options.mode == "folder":
        input_folder_path = Path(input_options.folder_path)
        if not input_folder_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder_path}")
        nd_file_list = sorted(input_folder_path.glob(input_options.glob_pattern))
    else:
        nd_file_list = [Path(file_path) for file_path in input_options.nd_file_list]

    if not nd_file_list:
        raise ValueError(
            f"No .nd files found in {input_folder_path} "
            f"with pattern {input_options.glob_pattern}"
            if input_options.mode == "folder"
            else "No .nd files provided in the list"
        )

    for nd_file_path in nd_file_list:
        if not nd_file_path.exists():
            raise FileNotFoundError(f"ND file not found: {nd_file_path}")

    tiles = []
    for nd_file_path in nd_file_list:
        logger.info(f"Generating tile list for {nd_file_path.name}")
        tiles.extend(tile_list_from_single_nd(nd_path=nd_file_path))

    logger.info(f"Aggregating {len(tiles)} tiles ...")
    tiled_images = tiles_aggregation_pipeline(
        tiles=tiles,
        converter_options=converter_options,
        filters=filters,
    )

    parallelization_list = setup_images_for_conversion(
        tiled_images=tiled_images,
        zarr_dir=zarr_dir,
        converter_options=converter_options,
        collection_type="SingleImage",
        overwrite_mode=OverwriteMode.OVERWRITE,
    )

    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=convert_visiview_nd_init_task,
        skip_logging_configuration=True,
    )
