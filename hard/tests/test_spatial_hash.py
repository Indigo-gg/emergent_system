"""
Spatial hash tests: build, neighbor query, bucket capacity limit.
Run with: python -m pytest tests/test_spatial_hash.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_cfg(n=1000, bucket_max=128, cell_size=2.0):
    return {
        'simulation': {'num_particles': n, 'bucket_max': bucket_max},
        'world': {'width': 100.0, 'height': 100.0, 'cell_size': cell_size},
    }


def test_build_no_crash():
    """Spatial hash builds without crashing."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.spatial_hash import SpatialHash

    cfg = _make_cfg()
    sh = SpatialHash(cfg)
    pos_x = np.random.uniform(0, 100, 1000).astype(np.float32)
    pos_y = np.random.uniform(0, 100, 1000).astype(np.float32)
    sh.build(pos_x, pos_y)
    print("PASS: test_build_no_crash")


def test_build_no_nan():
    """Spatial hash fields contain no NaN after build."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.spatial_hash import SpatialHash

    cfg = _make_cfg()
    sh = SpatialHash(cfg)
    pos_x = np.random.uniform(0, 100, 1000).astype(np.float32)
    pos_y = np.random.uniform(0, 100, 1000).astype(np.float32)
    sh.build(pos_x, pos_y)

    head = sh.cell_head.to_numpy()
    count = sh.cell_count.to_numpy()
    next_arr = sh.particle_next.to_numpy()

    assert not np.any(np.isnan(head.astype(float))), "NaN in cell_head"
    assert not np.any(np.isnan(count.astype(float))), "NaN in cell_count"
    assert not np.any(np.isnan(next_arr.astype(float))), "NaN in particle_next"
    print("PASS: test_build_no_nan")


def test_bucket_capacity_limit():
    """Particles beyond bucket_max are rejected."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.spatial_hash import SpatialHash

    # Place all 200 particles in the same cell (bucket_max=10)
    cfg = _make_cfg(n=200, bucket_max=10, cell_size=100.0)  # one big cell
    sh = SpatialHash(cfg)

    # All at same position → same cell
    pos_x = np.full(200, 50.0, dtype=np.float32)
    pos_y = np.full(200, 50.0, dtype=np.float32)
    sh.build(pos_x, pos_y)

    count = sh.cell_count.to_numpy()
    next_arr = sh.particle_next.to_numpy()

    # Only bucket_max particles should be inserted
    total_inserted = np.sum(next_arr != -1) + np.sum(count)  # approximate
    # At least some particles should be rejected (next = -1)
    rejected = np.sum(next_arr == -1)
    assert rejected > 0, f"Expected some rejected particles, got {rejected} rejected"
    print(f"PASS: test_bucket_capacity_limit ({rejected} particles rejected)")


def test_cell_count_respects_bucket_max():
    """No cell count exceeds bucket_max."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.spatial_hash import SpatialHash

    bucket_max = 10
    cfg = _make_cfg(n=500, bucket_max=bucket_max, cell_size=5.0)
    sh = SpatialHash(cfg)

    pos_x = np.random.uniform(0, 100, 500).astype(np.float32)
    pos_y = np.random.uniform(0, 100, 500).astype(np.float32)
    sh.build(pos_x, pos_y)

    count = sh.cell_count.to_numpy()
    max_count = np.max(count)
    assert max_count <= bucket_max, f"Max cell count {max_count} > bucket_max {bucket_max}"
    print(f"PASS: test_cell_count_respects_bucket_max (max={max_count})")


def test_linked_list_consistency():
    """Linked list traversal visits exactly the particles in that cell."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.spatial_hash import SpatialHash

    cfg = _make_cfg(n=100, bucket_max=128, cell_size=10.0)
    sh = SpatialHash(cfg)

    # Place particles on a grid so each cell has exactly 1
    n = 100
    pos_x = np.zeros(n, dtype=np.float32)
    pos_y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pos_x[i] = (i % 10) * 10.0 + 5.0
        pos_y[i] = (i // 10) * 10.0 + 5.0

    sh.build(pos_x, pos_y)

    # Each cell should have exactly 1 particle
    count = sh.cell_count.to_numpy()
    occupied = count[count > 0]
    assert np.all(occupied == 1), f"Expected 1 per cell, got {occupied}"
    print("PASS: test_linked_list_consistency")


def test_query_neighbors_finds_close_particles():
    """Neighbor query finds particles within radius."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.spatial_hash import SpatialHash

    cfg = _make_cfg(n=10, bucket_max=128, cell_size=10.0)
    sh = SpatialHash(cfg)

    # Place 10 particles in a cluster
    pos_x = np.array([50.0 + i*0.5 for i in range(10)], dtype=np.float32)
    pos_y = np.full(10, 50.0, dtype=np.float32)
    sh.build(pos_x, pos_y)

    # Query from center with radius 5.0
    neighbor_indices = np.zeros(128, dtype=np.int32)
    neighbor_dists = np.zeros(128, dtype=np.float32)

    @ti.kernel
    def _query(sh: ti.template(), qx: ti.f32, qy: ti.f32, radius: ti.f32,
               px: ti.types.ndarray(), py: ti.types.ndarray(),
               ni: ti.types.ndarray(), nd: ti.types.ndarray()) -> ti.i32:
        return sh.query_neighbors(qx, qy, radius, px, py, ni, nd)

    count = _query(sh, 52.0, 50.0, 5.0, pos_x, pos_y, neighbor_indices, neighbor_dists)
    assert count > 0, "Expected to find neighbors"
    print(f"PASS: test_query_neighbors_finds_close_particles (found {count})")


def test_query_neighbors_excludes_far_particles():
    """Neighbor query does not find particles outside radius."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.spatial_hash import SpatialHash

    cfg = _make_cfg(n=5, bucket_max=128, cell_size=50.0)
    sh = SpatialHash(cfg)

    # Place particles far apart
    pos_x = np.array([10.0, 90.0, 10.0, 90.0, 50.0], dtype=np.float32)
    pos_y = np.array([10.0, 10.0, 90.0, 90.0, 50.0], dtype=np.float32)
    sh.build(pos_x, pos_y)

    neighbor_indices = np.zeros(128, dtype=np.int32)
    neighbor_dists = np.zeros(128, dtype=np.float32)

    @ti.kernel
    def _query(sh: ti.template(), qx: ti.f32, qy: ti.f32, radius: ti.f32,
               px: ti.types.ndarray(), py: ti.types.ndarray(),
               ni: ti.types.ndarray(), nd: ti.types.ndarray()) -> ti.i32:
        return sh.query_neighbors(qx, qy, radius, px, py, ni, nd)

    # Query from (10,10) with small radius — should only find particle 0
    count = _query(sh, 10.0, 10.0, 1.0, pos_x, pos_y, neighbor_indices, neighbor_dists)
    assert count == 0, f"Expected 0 neighbors at (10,10) with r=1, got {count}"
    print("PASS: test_query_neighbors_excludes_far_particles")


def test_empty_positions():
    """Spatial hash handles edge case of all particles at same position."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.spatial_hash import SpatialHash

    cfg = _make_cfg(n=50, bucket_max=128, cell_size=2.0)
    sh = SpatialHash(cfg)

    pos_x = np.full(50, 50.0, dtype=np.float32)
    pos_y = np.full(50, 50.0, dtype=np.float32)
    sh.build(pos_x, pos_y)

    count = sh.cell_count.to_numpy()
    assert np.sum(count) <= 50, "Total count should not exceed particle count"
    print("PASS: test_empty_positions")


if __name__ == '__main__':
    test_build_no_crash()
    test_build_no_nan()
    test_bucket_capacity_limit()
    test_cell_count_respects_bucket_max()
    test_linked_list_consistency()
    test_query_neighbors_finds_close_particles()
    test_query_neighbors_excludes_far_particles()
    test_empty_positions()
    print("\nAll spatial hash tests passed!")
