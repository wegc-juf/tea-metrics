"""Helpers for drawing configured Austrian map boundary layers."""
from pathlib import Path

import geopandas as gpd
import yaml


def load_austrian_boundaries(config_file, target_crs, major_path=None, minor_path=None):
    """Load configured major and minor boundaries, reprojected to ``target_crs``."""
    config_file = Path(config_file)
    with config_file.open() as stream:
        config = yaml.safe_load(stream)

    configured_shape = config.get('create_region_masks', {}).get('shpfile')
    if not configured_shape:
        raise ValueError(f'No create_region_masks.shpfile configured in {config_file}')

    shape_dir = Path(configured_shape).parent
    layers = []
    for shape_path, filename, color, linewidth in (
            (major_path, 'LAND_AT_.shp', 'black', 0.9),
            (minor_path, 'GEM_AT_.shp', 'dimgray', 0.35)):
        shape_path = Path(shape_path) if shape_path is not None else shape_dir / filename
        if not shape_path.is_file():
            raise FileNotFoundError(f'Boundary shapefile not found: {shape_path}')
        boundaries = gpd.read_file(shape_path)
        if boundaries.crs is None:
            raise ValueError(f'Boundary shapefile has no CRS: {shape_path}')
        layers.append((boundaries.to_crs(target_crs), color, linewidth))

    return layers


def draw_austrian_boundaries(ax, layers):
    """Draw boundary geometries over an existing map axis."""
    for boundaries, color, linewidth in layers:
        boundaries.boundary.plot(ax=ax, color=color, linewidth=linewidth, zorder=3)
