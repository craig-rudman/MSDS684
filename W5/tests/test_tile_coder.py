import numpy as np
import pytest

from src.tile_coder import TileCoder


class TestTileCoderConstruction:
    def test_default_parameters(self):
        tc = TileCoder()
        assert tc.n_tilings == 8
        assert tc.tiles_per_dim == [8, 8]
        assert tc.state_bounds == [(-1.2, 0.6), (-0.07, 0.07)]

    def test_custom_parameters(self):
        tc = TileCoder(n_tilings=4, tiles_per_dim=[4, 4], state_bounds=[(-1.0, 1.0), (-0.5, 0.5)])
        assert tc.n_tilings == 4
        assert tc.tiles_per_dim == [4, 4]
        assert tc.state_bounds == [(-1.0, 1.0), (-0.5, 0.5)]

    def test_mutable_defaults_are_independent(self):
        tc1 = TileCoder()
        tc2 = TileCoder()
        tc1.tiles_per_dim.append(4)
        assert tc2.tiles_per_dim == [8, 8]

    def test_n_dims(self):
        assert TileCoder().n_dims == 2

    def test_tile_widths(self):
        tc = TileCoder()
        expected = np.array([(0.6 - (-1.2)) / 8, (0.07 - (-0.07)) / 8])
        np.testing.assert_array_almost_equal(tc.tile_widths, expected)

    def test_num_features(self):
        assert TileCoder().num_features == 8 * 8 * 8

    def test_num_features_custom(self):
        assert TileCoder(n_tilings=4, tiles_per_dim=[4, 4]).num_features == 4 * 4 * 4


class TestTileCoderGetTiles:
    @pytest.fixture
    def tc(self):
        return TileCoder()

    def test_returns_one_index_per_tiling(self, tc):
        assert len(tc.get_tiles([-0.5, 0.0])) == tc.n_tilings

    def test_indices_within_feature_range(self, tc):
        assert all(0 <= i < tc.num_features for i in tc.get_tiles([-0.5, 0.0]))

    def test_is_deterministic(self, tc):
        state = [-0.5, 0.0]
        assert tc.get_tiles(state) == tc.get_tiles(state)

    def test_different_states_produce_different_tiles(self, tc):
        assert tc.get_tiles([-0.5, 0.0]) != tc.get_tiles([0.0, 0.03])

    def test_at_lower_bound(self, tc):
        assert len(tc.get_tiles([-1.2, -0.07])) == tc.n_tilings

    def test_at_upper_bound(self, tc):
        assert len(tc.get_tiles([0.6, 0.07])) == tc.n_tilings
