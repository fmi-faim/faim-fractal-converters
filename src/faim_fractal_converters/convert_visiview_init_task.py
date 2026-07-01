from typing import Any

from loguru import logger
from ome2xarray import CompanionFile
from ome_zarr_converters_tools import (
    AcquisitionDetails,
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
    # TODO optionally sanitize metadata

    tiles: list[Tile] = []

    # each image index goes into a separate ome-zarr dataset
    for index, image in enumerate(ome.images):
        logger.info(f"Preparing image {index} ({image.stage_label.name})...")
        collection = SingleImage(
            image_path=f"s{int(image.stage_label.name.split(':')[0]) + 1}"
        )
        acquisition_details = (
            AcquisitionDetails()
        )  # TODO populate channel metadata etc.
        # loop over image planes and populate tiles at t and c level
        # (only once when z level is 0)
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
                    fov_name="default",
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
    run_fractal_task(task_function=convert_visiview_init_task, skip_logging_configuration=True)
