#%%
import os
import torch as t
import einops
import matplotlib.pyplot as plt
from ipywidgets import interact
import w1d1_test

MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")


def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    return t.Tensor([[[0,0,0], [1, y_limit*(-1+2*(k/(num_pixels-1))), 0]] for k in range(num_pixels)])


def render_lines_with_pyplot(lines: t.Tensor):
    """Plot any number of line segments in 3D.

    lines: shape (num_lines, num_points=2, num_dims=3).
    """
    (fig, ax) = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
    for line in lines:
        ax.plot(line[:, 1].numpy(), line[:, 0].numpy(), line[:, 2].numpy())
    ax.set(xlabel="Y", ylabel="X", zlabel="Z")
    return fig


rays1d = make_rays_1d(9, 10.0)
if MAIN and (not IS_CI):
    render_lines_with_pyplot(rays1d)

#%%

import os
import torch as t
import einops
import matplotlib.pyplot as plt
from ipywidgets import interact
import w1d1_test

@interact
def line(v=(-2.0, 2.0), seed=(0, 10)):
    """
    Interactive line widget.

    Drag "seed" to get a different random line.
    Drag "v" to see that v must be in [0, 1] for the intersection marked by a star to be "inside" the object line.
    """
    t.manual_seed(seed)
    L_1 = t.randn(2)
    L_2 = t.randn(2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    (x, y) = zip(P(-2), P(2))
    plt.plot(x, y, "g-")
    plt.plot(*L_1, "ro", markersize=12)
    plt.plot(*L_2, "ro", markersize=12)
    plt.plot(P(v)[0], P(v)[1], "*", markersize=12)
    plt.xlabel("X")
    plt.ylabel("Y")

#%%

import os
import torch as t
import einops
import matplotlib.pyplot as plt
from ipywidgets import interact
import w1d1_test
import numpy

def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    O, D = ray[0], ray[1]
    L_1, L_2 = segment[0], segment[1]

    A = t.Tensor([[D[0].item(), (L_1-L_2)[0].item()], [D[1].item(), (L_1-L_2)[1].item()]])
    B = t.Tensor([(L1 - O)[0].item(), (L1 - O)[1].item()])

    if t.linalg.det(A) == 0:
        return False

    mytensor = t.linalg.solve(A,B)

    if (mytensor[0].item() >= 0 and mytensor[1].item() >= 0 and mytensor[1].item() <= 1):
        return True

    else:
        return False

w1d1_test.test_intersect_ray_1d(intersect_ray_1d)
w1d1_test.test_intersect_ray_1d_special_case(intersect_ray_1d)

#%%