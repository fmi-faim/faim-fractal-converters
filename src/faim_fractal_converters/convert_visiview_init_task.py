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


def convert_visiview_init_task(
        *,
        # Fractal parameters
        zarr_dir: str,
        # Task parameters
        companion_ome_file: str,
        sanitize_metadata: bool = True,
        converter_options: ConverterOptions = ConverterOptions()
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
            image_path=f"s{int(image.stage_label.name.split(':')[0])+1}"
        )
        acquisition_details = AcquisitionDetails()  # TODO populate channel metadata etc.
        # loop over image planes and populate tiles at t and c level
        # (only once when z level is 0)
        tiff_block_dict = {}
        for block in image.pixels.tiff_data_blocks:
            if block.first_z == 0:
                tiff_block_dict[(block.first_t, block.first_c)] = block.uuid.file_name
        for plane in image.pixels.planes:
            if plane.the_z == 0:
                tiles.append(
                    Tile(
                        fov_name="default",
                        start_x=plane.position_x,
                        start_y=plane.position_y,
                        start_z=plane.position_z,
                        start_c=plane.the_c,
                        start_t=plane.the_t,
                        length_x=image.pixels.size_x,
                        length_y=image.pixels.size_y,
                        length_z=image.pixels.size_z,
                        length_c=1,
                        length_t=1,
                        attributes={},
                        collection=collection,
                        image_loader=DefaultImageLoader(
                            file_path=(companion._data_folder / tiff_block_dict[(plane.the_t, plane.the_c)]).as_posix(),
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
