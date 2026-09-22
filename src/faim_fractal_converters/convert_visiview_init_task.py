from pathlib import Path
from typing import Annotated, Any, Literal

from loguru import logger
from ome2xarray import CompanionFile
from ome_types.model import Channel
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

from faim_fractal_converters.utils import reverse_mapping

DefaultConverterOptions = ConverterOptions()


class FolderLoading(BaseModel):
    """Load companion.ome files from folder."""

    mode: Literal["folder"] = "folder"
    folder_path: str
    glob_pattern: str = "*.companion.ome"


class ListLoading(BaseModel):
    """Load a list of specified companion.ome files."""

    mode: Literal["list"] = "list"
    ome_file_list: list[str]


Loading = Annotated[
    FolderLoading | ListLoading,
    Field(discriminator="mode"),
]


class SanitizingOptions(BaseModel):
    channel_mapping: dict[int, str] | None = None
    tf2_extension: bool = False


def _color_RGBA_int_to_RGB_hex(color: int) -> str:
    r = (color >> 24) & 0xFF
    g = (color >> 16) & 0xFF
    b = (color >> 8) & 0xFF
    return f"#{r:02X}{g:02X}{b:02X}"


def _generate_channel_info(channel: Channel):
    return ChannelInfo(
        channel_label=channel.name,
        wavelength_id=str(channel.excitation_wavelength),
        color=None
        if int(channel.color) == -1
        else _color_RGBA_int_to_RGB_hex(int(channel.color)),
    )


def _generate_image_naming(image: Any) -> tuple[str, str]:
    """Get (Path, FOV name) from image metadata."""
    stage_label = getattr(image, "stage_label", None)
    if stage_label is None:
        return image.name, "default"

    stage_label_name = stage_label.name
    image_name = image.name.removesuffix(f"_{stage_label_name}")
    stage_index = int(stage_label_name.split(":", maxsplit=1)[0]) + 1

    sg_suffix = stage_label_name.rpartition("_")[2]
    if sg_suffix.startswith("sg:"):
        image_name = f"{image_name}_sg{int(sg_suffix.split(':', maxsplit=1)[1])}"

    return image_name, f"s{stage_index}"


def tile_list_from_single_ome(
    companion_ome_path: Path,
    sanitizing_options: SanitizingOptions | None = None,
) -> list[Tile]:
    companion = CompanionFile(companion_ome_path)
    ome = companion.get_ome_metadata()
    # optionally sanitize metadata
    if sanitizing_options is not None:
        for index, image in enumerate(ome.images):
            companion.sanitize_image(
                image_index=index,
                channel_mapping=sanitizing_options.channel_mapping,
                big_tiff=sanitizing_options.tf2_extension,
            )
        ome = companion.get_ome_metadata()

    tiles: list[Tile] = []
    logger.info(f"This companion file defines {len(ome.images)} images...")
    for index, image in enumerate(ome.images):
        image_path, fov_name = _generate_image_naming(image)
        collection = SingleImage(image_path=image_path)
        acquisition_details = AcquisitionDetails(
            start_t_space="pixel",
            xy_pixel_size=image.pixels.physical_size_x,  # we only support isotropic xy values
            z_spacing=image.pixels.physical_size_z or 1.0,
            t_spacing=image.pixels.time_increment or 1.0,
            channels=[
                _generate_channel_info(channel) for channel in image.pixels.channels
            ],
        )

        tiff_block_dict = {}
        for block in image.pixels.tiff_data_blocks:
            tiff_block_dict[(block.first_t, block.first_c, block.first_z)] = (
                block.uuid.file_name
            )
        tifffile_to_plane_mapping = reverse_mapping(
            tcz_to_file=tiff_block_dict,
            assert_unique_tc=True,
            assert_contiguous_z=True,
        )
        plane_position_dict = {}
        for plane in image.pixels.planes:
            plane_position_dict[(plane.the_t, plane.the_c, plane.the_z)] = (
                plane.position_x,
                plane.position_y,
                plane.position_z,
            )
        for filename, tc_map in tifffile_to_plane_mapping.items():
            assert len(tc_map) == 1, (
                f"File {filename} maps to multiple (t, c) pairs: {sorted(tc_map)}"
            )
            # Add Tile for current file and (t, c) pair, using the first z value to get the position
            (t, c), z_list = next(iter(tc_map.items()))
            first_z = z_list[0]
            tiles.append(
                Tile(
                    fov_name=fov_name,
                    start_x=plane_position_dict[(t, c, first_z)][0],
                    start_y=plane_position_dict[(t, c, first_z)][1],
                    start_z=plane_position_dict[(t, c, first_z)][2],
                    start_c=c,
                    start_t=t,
                    length_x=image.pixels.size_x,
                    length_y=image.pixels.size_y,
                    length_z=len(z_list),
                    length_c=1,
                    length_t=1,
                    attributes={},
                    collection=collection,
                    image_loader=DefaultImageLoader(
                        file_path=(companion._data_folder / filename).as_posix(),
                    ),
                    acquisition_details=acquisition_details,
                )
            )
    return tiles


@validate_call
def convert_visiview_init_task(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Task parameters
    # companion_ome_file: str,
    input_options: Loading,
    sanitizing_options: SanitizingOptions | None = None,
    converter_options: ConverterOptions = DefaultConverterOptions,
    filters: list[ImplementedFilters] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Initialization task: parse acquisition and build parallelization list."""
    converter_options = ConverterOptions.model_validate(converter_options)

    # Folder loading mode
    if input_options.mode == "folder":
        input_folder_path = Path(input_options.folder_path)
        if not input_folder_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder_path}")
        ome_file_list = sorted(input_folder_path.glob(input_options.glob_pattern))

    # List loading mode
    elif input_options.mode == "list":
        ome_file_list = [Path(f) for f in input_options.ome_file_list]

    # Ensure non-empty file list
    if not ome_file_list:
        raise ValueError(
            f"No companion OME files found in {input_folder_path} "
            f"with pattern {input_options.glob_pattern}"
            if input_options.mode == "folder"
            else "No companion OME files provided in the list"
        )

    # Ensure all input files exist
    for ome_file_path in ome_file_list:
        if not ome_file_path.exists():
            raise FileNotFoundError(f"Companion OME file not found: {ome_file_path}")

    tiles = []
    for companion_ome_path in ome_file_list:
        logger.info(f"Generating tile list for {companion_ome_path.name}")
        tiles.extend(
            tile_list_from_single_ome(
                companion_ome_path=companion_ome_path,
                sanitizing_options=sanitizing_options,
            )
        )

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
        task_function=convert_visiview_init_task, skip_logging_configuration=True
    )
