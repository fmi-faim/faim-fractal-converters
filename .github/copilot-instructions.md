# Copilot instructions

## Project overview

- This is a Python package that provides Fractal converter tasks for Visiview `companion.ome` inputs.
- Source code lives under `src/faim_fractal_converters/` and uses a Fractal-style split between:
  - `convert_visiview_init_task.py` for parsing metadata and building the parallelization list
  - `dev/task_list.py` for task registration
- `src/faim_fractal_converters/__FRACTAL_MANIFEST__.json` is generated metadata for the Fractal task package and should stay aligned with the task definitions.

## Commands

Use Pixi for everything; the host Python may not match the project requirement.

- Full test suite: `pixi run --environment dev test`
- Single test: `pixi run --environment dev pytest tests/test_convert_visiview_init_task.py -q`
- Lint: `pixi run --environment dev ruff check src tests`
- Regenerate the Fractal manifest: `pixi run --environment dev create-manifest`

## Architecture

- The init task reads a Visiview companion file with `ome2xarray.CompanionFile`, extracts OME metadata, builds `Tile` objects, runs `tiles_aggregation_pipeline`, and returns `{"parallelization_list": ...}` for Fractal.

## Conventions

- Keep Fractal task entry points as top-level functions with keyword-only parameters.
- Preserve `@validate_call` on task functions when present.
- Use `pathlib.Path` for file and directory handling.
- Keep imports guarded with `TYPE_CHECKING` when they are only needed for annotations.
