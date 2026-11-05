"""Helpers for drawing configured Austrian map boundary layers."""
from pathlib import Path

import geopandas as gpd
import yaml


def load_austrian_boundaries(config_file, target_crs, major_path=None, minor_path=None):
    """Load selected boundaries and reproject them to ``target_crs``.

    With neither path selected, load both configured defaults. If either path
    is selected, load only the selected layers. A value of ``True`` selects
    that layer's configured default path; a path selects a custom shapefile.
    """
    config_file = Path(config_file)
    with config_file.open() as stream:
        config = yaml.safe_load(stream)

    configured_shape = config.get('create_region_masks', {}).get('shpfile')
    if not configured_shape:
        raise ValueError(f'No create_region_masks.shpfile configured in {config_file}')

    shape_dir = Path(configured_shape).parent
    selection_made = major_path is not None or minor_path is not None
    layer_specs = (
        (minor_path, 'GEM_AT_.shp', 'dimgray', 0.35, 2),
        (major_path, 'LAND_AT_.shp', 'black', 0.9, 4),
    )
    layers = []
    for selected_path, filename, color, linewidth, zorder in layer_specs:
        if selection_made and selected_path is None:
            continue
        shape_path = (shape_dir / filename if selected_path is None or selected_path is True
                      else Path(selected_path))
        if not shape_path.is_file():
            raise FileNotFoundError(f'Boundary shapefile not found: {shape_path}')
        boundaries = gpd.read_file(shape_path)
        if boundaries.crs is None:
            raise ValueError(f'Boundary shapefile has no CRS: {shape_path}')
        layers.append((boundaries.to_crs(target_crs), color, linewidth, zorder))

    return layers


def draw_austrian_boundaries(ax, layers):
    """Draw boundary geometries over an existing map axis."""
    for boundaries, color, linewidth, zorder in layers:
        boundaries.boundary.plot(ax=ax, color=color, linewidth=linewidth, zorder=zorder)
