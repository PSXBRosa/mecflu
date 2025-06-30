import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
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
        n = (y - y_min) / L
        ux[0, j] = u_in_max * (1 - n**2)
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


def create_laplacian_operator(nx, ny, dx, dy):
    """Creates a sparse Laplacian operator D for a scalar field."""
    N = nx * ny
    D = sparse.lil_matrix((N, N))
    
    # Iterate over interior grid points
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            k = i * ny + j
            D[k, k] = -2 * (1/dx**2 + 1/dy**2)
            D[k, k - ny] = 1/dx**2  # West
            D[k, k + ny] = 1/dx**2  # East
            D[k, k - 1] = 1/dy**2   # South
            D[k, k + 1] = 1/dy**2   # North
            
    return D.tocsr()

def create_gradient_operators(nx, ny, dx, dy):
    """Creates sparse gradient operators Gx and Gy."""
    N_p = (nx - 1) * (ny - 1) # Number of pressure points 
    N_u = nx * ny             # Number of velocity points 

    Gx = sparse.lil_matrix((N_u, N_p))
    Gy = sparse.lil_matrix((N_u, N_p))

    for i in range(nx - 1):
        for j in range(ny - 1):
            k_p = i * (ny - 1) + j
            
            # Approximate pressure 
            k_sw = i * ny + j      
            k_se = i * ny + (j + 1) 
            k_nw = (i + 1) * ny + j  
            k_ne = (i + 1) * ny + (j+1) 

            # Gx
            Gx[k_sw, k_p] = -1/dx
            Gx[k_se, k_p] = -1/dx
            Gx[k_nw, k_p] = 1/dx
            Gx[k_ne, k_p] = 1/dx
            
            # Gy
            Gy[k_sw, k_p] = -1/dy
            Gy[k_nw, k_p] = -1/dy
            Gy[k_se, k_p] = 1/dy
            Gy[k_ne, k_p] = 1/dy

    return Gx.tocsr(), Gy.tocsr()

def create_convection_operator(nx, ny, dx, dy, u_flat, v_flat):
    """Creates a sparse convection operator C based on upwinding."""
    N = nx * ny
    C = sparse.lil_matrix((N, N))
    
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            k = i * ny + j
            
            # Velocities at the current point
            u, v = u_flat[k], v_flat[k]
            
            # X-convection (upwind)
            if u > 0:
                C[k, k] += u / dx
                C[k, k - ny] -= u / dx
            else:
                C[k, k] -= u / dx
                C[k, k + ny] += u / dx
                
            # Y-convection (upwind)
            if v > 0:
                C[k, k] += v / dy
                C[k, k - 1] -= v / dy
            else:
                C[k, k] -= v / dy
                C[k, k + 1] += v / dy
    
    return C.tocsr()


def solve_navier_stokes_implicit(grid, dt, T, nu):
    xs, ys, ux, uy, x_centers, y_centers, p = grid
    nx, ny = ux.shape
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    
    # Get operator matrices
    D = create_laplacian_operator(nx, ny, dx, dy)
    Gx, Gy = create_gradient_operators(nx, ny, dx, dy)
    Div = -sparse.bmat([[Gx.T, Gy.T]]) # Divergence is -G.T

    # Time-stepping
    t = 0
    while t < T:
        print(f"Solving at t = {t:.4f}")
        
        # Flatten velocity vectors for matrix operations
        u_n = ux.flatten()
        v_n = uy.flatten()

        # 1. Assemble the monolithic matrix A
        C = create_convection_operator(nx, ny, dx, dy, u_n, v_n)
        
        A_u = (1/dt) * sparse.eye(nx * ny) + C + nu * D
        A_v = (1/dt) * sparse.eye(nx * ny) + C + nu * D
        
        A = sparse.bmat([
            [A_u,   None,     -Gx],
            [None,  A_v,      -Gy],
            [Div,   None,     None]
        ], format='csr')

        # 2. Assemble the right-hand side vector b
        b_u = u_n / dt
        b_v = v_n / dt
        b_p = np.zeros(p.size)
        b = np.concatenate([b_u, b_v, b_p])

        # 3. Apply Boundary Conditions
        # TODO
        
        # 4. Solve the linear system Ax = b
        try:
            x = spsolve(A, b)
        except Exception as e:
            print(f"Solver failed: {e}")
            return ux, uy, p

        # 5. Unpack the solution
        u_new = x[:nx*ny]
        v_new = x[nx*ny : 2*nx*ny]
        p_new = x[2*nx*ny:]

        ux = u_new.reshape((nx, ny))
        uy = v_new.reshape((nx, ny))
        p = p_new.reshape(p.shape)
        
        # Update for next step and apply boundary conditions to the fields
        apply_boundary((xs, ys, ux, uy, x_centers, y_centers, p))
        
        t += dt
        
    return ux, uy, p


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
    Nx = 40  # Number of nodes in x-direction
    Ny = 20  # Number of nodes in y-direction
    refining_factor = 2.0

    # Simulation parameters
    dt = 0.01      # time step
    T = 0.5        # total time
    nu = 0.01      # kinematic viscosity

    grid = create_grid(Nx, Ny, refining_factor)
    apply_boundary(grid)

    # Initial plot of the grid can be useful
    # plot_grid(grid) 

    # Run the implicit solver
    ux, uy, p = solve_navier_stokes_implicit(grid, dt, T, nu)

    # Visualize the final results
    plot_results(grid[0], grid[1], ux, uy, p) # Assuming you have a plot_results function