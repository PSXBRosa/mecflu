import numpy as np
import matplotlib.pyplot as plt


def get_neighbors(i, j, nx, ny):
    for di, dj in ((-1,0),(1,0),(0,-1),(0,1)):
        ni, nj = i+di, j+dj
        if 0 <= ni < nx and 0 <= nj < ny:
            yield ni, nj
    

def create_grid(Nx, Ny, refining_factor=2):
    """
    Create a grid with a refined central region around a pipe.
    The grid is defined in a rectangular domain with a pipe in the center.

    Parameters:
    Nx : int
        Number of nodes in the x-direction.
    Ny : int
        Number of nodes in the y-direction.
    refining_factor : float
        Factor to control the number of nodes in the refined region.

    Returns:
    xs : np.ndarray
        x-coordinates of the grid nodes.
    ys : np.ndarray
        y-coordinates of the grid nodes.
    ux : np.ndarray
        x-component of the velocity at the grid nodes.
    uy : np.ndarray
        y-component of the velocity at the grid nodes.
    x_centers : np.ndarray
        x-coordinates of the cell centers.
    y_centers : np.ndarray
        y-coordinates of the cell centers.
    p : np.ndarray
        Pressure at the grid centers.
    """
    D = 1/5  

    x_min, x_max = 0, 20 * D
    y_min, y_max = 0, 5 * D
    x_pipe, y_pipe = 5 * D, 2.5 * D
    
    # Largura da região de refinamento 
    w = 2*D
    
    # Número de pontos na região central refinada
    fraction = 2 * w / (x_max - x_min)
    nx_center = int(refining_factor * Nx * fraction)
    nx_center = max(1, min(nx_center, Nx - 2))
    
    # Pontos nas regiões lateral, central e lateral direita
    nx_side = (Nx - nx_center) // 2
    xs_left = np.linspace(x_min, x_pipe - w, nx_side + 1)[:-1]
    xs_center = np.linspace(x_pipe - w, x_pipe + w, nx_center + 1)[:-1]
    xs_right = np.linspace(x_pipe + w, x_max, Nx - len(xs_left) - len(xs_center))

    xs = np.concatenate([xs_left, xs_center, xs_right])
    ys = np.linspace(y_min, y_max, Ny + 1)

    ux = np.zeros((len(xs), len(ys)))
    uy = np.zeros((len(xs), len(ys)))

    x_centers = (xs[:-1] + xs[1:]) / 2
    y_centers = (ys[:-1] + ys[1:]) / 2

    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')

    p = np.zeros_like(Xc)

    return xs, ys, ux, uy, x_centers, y_centers, p


def apply_boundary(grid, u_in_max=1.0):
    xs, ys, ux, uy, x_centers, y_centers, p = grid
    nx, ny = ux.shape

    y_min, y_max = ys[0], ys[-1]
    L = y_max - y_min

    # --- 1) Inlet: x = xs[0] (i = 0) ---
    for j, y in enumerate(ys):
        η = (y - y_min) / L
        ux[0, j] = u_in_max * (1 - η**2)
        uy[0, j] = 0.0

    # --- 2) Walls: y = ys[0] e ys[-1] (j = 0 e j = ny-1) ---
    ux[:, 0]  = 0.0
    uy[:, 0]  = 0.0
    ux[:, -1] = 0.0
    uy[:, -1] = 0.0

    # --- 3) Outlet: x = xs[-1] (i = nx-1) zero‐gradient ---
    ux[-1, :] = ux[-2, :]
    uy[-1, :] = uy[-2, :]

    # --- 4) Pression: zero‐gradient ---
    p[ 0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    p[:,  0] = p[:, 1]
    p[:, -1] = p[:, -2]


def discretize_space(grid):
    xs, ys, ux, uy, x_centers, y_centers, p = grid
    nx, ny = ux.shape
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # upwind in x for first derivatives
    dudx = np.zeros_like(ux)
    dvdx = np.zeros_like(uy)
    u_center = ux[1:-1, :]

    dudx[1:-1, :] = np.where(
        u_center > 0,
        ( u_center      - ux[:-2, :] ) / dx,
        ( ux[2:, :]     - u_center    ) / dx
    )
    dvdx[1:-1, :] = np.where(
        u_center > 0,
        ( uy[1:-1, :]   - uy[:-2, :] ) / dx,
        ( uy[2:, :]     - uy[1:-1, :] ) / dx
    )
    # one‐sided at boundaries
    dudx[0,  :]  = (ux[1,  :]  - ux[0,  :]) / dx
    dudx[-1, :]  = (ux[-1, :]  - ux[-2, :]) / dx
    dvdx[0,  :]  = (uy[1,  :]  - uy[0,  :]) / dx
    dvdx[-1, :]  = (uy[-1, :]  - uy[-2, :]) / dx

    # central in y for first derivatives
    dudy = np.zeros_like(ux)
    dvdy = np.zeros_like(uy)
    dudy[:, 1:-1] = (ux[:, 2:] - ux[:, :-2]) / (2*dy)
    dvdy[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) / (2*dy)

    # second derivatives (unchanged)
    d2udx2 = np.zeros_like(ux)
    d2udy2 = np.zeros_like(ux)
    d2vdx2 = np.zeros_like(uy)
    d2vdy2 = np.zeros_like(uy)

    d2udx2[1:-1, :] = (ux[2:, :] - 2*ux[1:-1, :] + ux[:-2, :]) / dx**2
    d2udy2[:, 1:-1] = (ux[:, 2:] - 2*ux[:, 1:-1] + ux[:, :-2]) / dy**2
    d2vdx2[1:-1, :] = (uy[2:, :] - 2*uy[1:-1, :] + uy[:-2, :]) / dx**2
    d2vdy2[:, 1:-1] = (uy[:, 2:] - 2*uy[:, 1:-1] + uy[:, :-2]) / dy**2

    # convective terms
    conv_u = ux * dudx + uy * dudy
    conv_v = ux * dvdx + uy * dvdy

    return dudx, dudy, dvdx, dvdy, \
           d2udx2, d2udy2, d2vdx2, d2vdy2, \
           conv_u, conv_v


def discretize_time(discretized_space):
    # TODO
    pass


def plot_grid(grid):
    xs, ys, ux, uy, x_centers, y_centers, p = grid
    
    plt.figure(figsize=(10, 5))
    plt.scatter(xs, ys, c='blue', marker='o')
    plt.title('Grid Nodes')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    Nx = 60 # Number of nodes in x-direction
    Ny = 30 # Number of nodes in y-direction
    refining_factor = 3.0  # Adjust this for grid refinement

    grid = create_grid(Nx, Ny, refining_factor)
    plot_grid(grid)

    # Print the grid nodes
    for node in grid:
        print(node)