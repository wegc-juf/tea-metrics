"""Helpers for stable filenames for GRG-grid-dependent products."""


def _format_grid_value(value):
    """Format a numeric grid value consistently, regardless of int/float input."""
    return format(float(value), '.12g').replace('.', 'p')


def grg_grid_token(opts):
    """Return a filename token containing both GRG spacing and cell size."""
    spacing = getattr(opts, 'grg_grid_spacing', 0.5)
    default_cell_size = 1 if getattr(opts, 'precip', False) else 2
    cell_size = getattr(opts, 'grg_cell_size', default_cell_size)
    return (f'{_format_grid_value(spacing)}_'
            f'cell{_format_grid_value(cell_size)}')
