from typing import Any

from loguru import logger
from ome2xarray import CompanionFile
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    ChannelInfo,
    ConverterOptions,
    DefaultImageLoader,
    OverwriteMode,
    SingleImage,
    Tile,
    setup_images_for_conversion,
    tiles_aggregation_pipeline,
)
from pathlib import Path
from faim_fractal_converters.utils import reverse_mapping

from ome_types.model import Channel


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


def convert_visiview_init_task(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Task parameters
    companion_ome_file: str,
    sanitize_metadata: bool = True,
    converter_options: ConverterOptions = ConverterOptions(),
) -> dict[str, list[dict[str, Any]]]:
    """Initialization task: parse acquisition and build parallelization list."""
    companion = CompanionFile(Path(companion_ome_file))
    ome = companion.get_ome_metadata()
    # optionally sanitize metadata
    if sanitize_metadata:
        for index, image in enumerate(ome.images):
            companion.sanitize_image(image_index=index)
        ome = companion.get_ome_metadata()

    tiles: list[Tile] = []
    converter_options = ConverterOptions.model_validate(converter_options)

    # each image index goes into a separate ome-zarr dataset
    for index, image in enumerate(ome.images):
        image_path, fov_name = _generate_image_naming(image)
        logger.info(f"Preparing image {index} ({image_path})...")
        collection = SingleImage(image_path=image_path)
        acquisition_details = AcquisitionDetails(
            xy_pixel_size=image.pixels.physical_size_x,  # we only support isotropic xy values
            z_spacing=image.pixels.physical_size_z,
            t_spacing=image.pixels.time_increment,
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

    tiled_images = tiles_aggregation_pipeline(
        tiles=tiles,
        converter_options=converter_options,
    )

    parallelization_list = setup_images_for_conversion(
        tiled_images=tiled_images,
        zarr_dir=zarr_dir,
        converter_options=converter_options,
        collection_type="SingleImage",
        overwrite_mode=OverwriteMode.OVERWRITE,
        ngff_version="0.5",
    )

    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=convert_visiview_init_task, skip_logging_configuration=True
    )
