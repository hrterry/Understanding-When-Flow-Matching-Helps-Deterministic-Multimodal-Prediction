import numpy as np
from scipy.spatial import KDTree

from .distribution_utils import get_distribution


class PatchSampler:
    def __init__(
        self,
        distribution: str = 'batch_128',
        min_samples=2,
        mode: str = "local",
        max_samples: int | None = None,
    ):
        self.distribution = distribution
        self.distribution_func = get_distribution(distribution)
        self.min_samples = min_samples
        self.mode = mode
        self.max_samples = None if max_samples is None else int(max_samples)
        if self.max_samples is not None and self.max_samples <= 0:
            self.max_samples = None

        if self.mode not in {"local", "global"}:
            raise ValueError(f"Unknown sampling mode: {self.mode}. Expected one of ['local', 'global'].")

    def sample_nearest_patch(self, coords, num_samples):
        num_samples = min(len(coords), num_samples)

        if num_samples == len(coords):
            return np.arange(len(coords))

        tree = KDTree(coords)
        center_idx = np.random.randint(0, len(coords))
        center_coord = coords[center_idx]
        _, idx_nearest = tree.query(center_coord, k=num_samples)
        return idx_nearest

    def sample_global_patch(self, coords, num_samples):
        num_samples = min(len(coords), num_samples)

        if num_samples == len(coords):
            return np.arange(len(coords))
        return np.random.choice(len(coords), size=num_samples, replace=False)

    def get_distribution_expectation(self):
        return np.mean([self.distribution_func() for _ in range(10000)])

    def __call__(self, coords):
        total_samples = max(self.min_samples, int(len(coords) * self.distribution_func()))
        if self.max_samples is not None:
            total_samples = min(total_samples, self.max_samples)
            total_samples = max(1, total_samples)
        if self.mode == "local":
            return self.sample_nearest_patch(coords, total_samples)
        return self.sample_global_patch(coords, total_samples)


if __name__ == "__main__":
    # Test the patch sampler
    coords = np.random.rand(100, 2)
    sampler = PatchSampler("beta_3_1")
    print(sampler(coords).shape)
    print(sampler.get_distribution_expectation())