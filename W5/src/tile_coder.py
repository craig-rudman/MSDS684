from __future__ import annotations

import math
import numpy as np


class TileCoder:
    """Maps continuous observations to active tile indices via overlapping tilings."""

    def __init__(
        self,
        n_tilings: int = 8,
        tiles_per_dim: list[int] | None = None,
        state_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim if tiles_per_dim is not None else [8, 8]
        self.state_bounds = state_bounds if state_bounds is not None else [(-1.2, 0.6), (-0.07, 0.07)]
        self.n_dims = len(self.tiles_per_dim)
        self.tile_widths = np.array([
            (high - low) / n
            for (low, high), n in zip(self.state_bounds, self.tiles_per_dim)
        ])
        self.num_features = self.n_tilings * math.prod(self.tiles_per_dim)

    def get_tiles(self, state: list[float]) -> list[int]:
        """Return the flat tile index for each tiling given a continuous state."""
        indices = []
        tiles_per_tiling = math.prod(self.tiles_per_dim)

        for t in range(self.n_tilings):
            # Stagger each tiling by a fraction of one tile width per dimension.
            # This ensures tilings overlap asymmetrically, giving smooth generalization
            # across the state space.
            offsets = np.array([t * self.tile_widths[d] / self.n_tilings
                                 for d in range(self.n_dims)])

            coords = []
            for d in range(self.n_dims):
                low = self.state_bounds[d][0]
                coord = int((state[d] - low + offsets[d]) / self.tile_widths[d])
                coord = max(0, min(coord, self.tiles_per_dim[d] - 1))
                coords.append(coord)

            flat_idx = 0
            stride = 1
            for d in reversed(range(self.n_dims)):
                flat_idx += coords[d] * stride
                stride *= self.tiles_per_dim[d]

            indices.append(t * tiles_per_tiling + flat_idx)

        return indices

