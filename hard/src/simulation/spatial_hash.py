"""
GPU-native spatial hash with bucket capacity hard limit.

Uses cell_head + particle_next linked-list scheme with atomic operations.
Bucket capacity hard limit prevents O(N^2) collapse when particles clump.
"""

import taichi as ti


@ti.data_oriented
class SpatialHash:
    """GPU spatial hash grid with bucket capacity limit."""

    def __init__(self, cfg: dict):
        self.world_w = cfg['world']['width']
        self.world_h = cfg['world']['height']
        self.cell_size = cfg['world']['cell_size']
        self.bucket_max = cfg['simulation']['bucket_max']
        self.n_particles = cfg['simulation']['num_particles']

        self.cols = int(self.world_w / self.cell_size) + 1
        self.rows = int(self.world_h / self.cell_size) + 1
        self.num_cells = self.cols * self.rows

        # Linked-list structure
        self.cell_head = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.particle_next = ti.field(dtype=ti.i32, shape=self.n_particles)
        self.cell_count = ti.field(dtype=ti.i32, shape=self.num_cells)

    @ti.kernel
    def build(self, pos_x: ti.template(), pos_y: ti.template()):
        """Build spatial hash from current particle positions.

        Accepts Taichi fields (GPU, zero-copy) directly — no CPU-GPU transfer.
        Previously accepted numpy arrays which forced a to_numpy() round-trip.
        """
        # Reset
        for cell in range(self.num_cells):
            self.cell_head[cell] = -1
            self.cell_count[cell] = 0

        # Insert particles with bucket capacity limit
        for i in range(self.n_particles):
            col = int(pos_x[i] / self.cell_size)
            row = int(pos_y[i] / self.cell_size)
            # Clamp to valid range
            col = ti.math.clamp(col, 0, self.cols - 1)
            row = ti.math.clamp(row, 0, self.rows - 1)
            cell = row * self.cols + col

            count = ti.atomic_add(self.cell_count[cell], 1)
            if count < self.bucket_max:
                # Insert at head of linked list
                old = self.cell_head[cell]
                self.cell_head[cell] = i
                self.particle_next[i] = old
            else:
                # Bucket full — particle excluded from neighbor queries this step
                self.particle_next[i] = -1

    @ti.func
    def query_neighbors(self, qx: ti.f32, qy: ti.f32, radius: ti.f32,
                        pos_x: ti.types.ndarray(), pos_y: ti.types.ndarray(),
                        neighbor_indices: ti.types.ndarray(),
                        neighbor_dists: ti.types.ndarray()) -> ti.i32:
        """
        Find all particles within radius of (qx, qy).
        Returns count. Results written to neighbor_indices and neighbor_dists.
        """
        r2 = radius * radius
        count = 0

        min_col = ti.math.clamp(int((qx - radius) / self.cell_size), 0, self.cols - 1)
        max_col = ti.math.clamp(int((qx + radius) / self.cell_size), 0, self.cols - 1)
        min_row = ti.math.clamp(int((qy - radius) / self.cell_size), 0, self.rows - 1)
        max_row = ti.math.clamp(int((qy + radius) / self.cell_size), 0, self.rows - 1)

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell = row * self.cols + col
                idx = self.cell_head[cell]
                while idx != -1:
                    dx = qx - pos_x[idx]
                    dy = qy - pos_y[idx]
                    d2 = dx * dx + dy * dy
                    if d2 < r2 and d2 > 0.0:
                        if count < 128:  # max neighbors per query
                            neighbor_indices[count] = idx
                            neighbor_dists[count] = ti.sqrt(d2)
                            count += 1
                    idx = self.particle_next[idx]

        return count
