# euler1D.py
"""ROMs for 1D lifted Euler equations."""

import os
import h5py
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.integrate as sin
import scipy.sparse as sparse
import scipy.interpolate as interp

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.animation as animation
from IPython.display import HTML

import rom_operator_inference as opinf

import utils


# Helper functions ============================================================

def _extend_time(t, factor):
    dt = t[1] - t[0]
    return np.arange(t[0], factor*(t[-1] - t[0]) + t[0] + dt, dt)


# Solver classes ==============================================================

class EulerSolver:
    """Bundles a high-fidelity solver, data management, and plotting tools
    for the one-dimensional Euler equations with periodic boundary conditions.

    ROM learning is implemented by child classes.

    Attributes
    ----------
    snapshots : list of s (3n,k) ndarrays
        State snapshots, one for each initial condition.
        Variables stored are [u, p, 1/rho].
    noise : list of s (3n,k) ndarray
        Gaussian noise to be applied to the conservative state variables
        (see apply_noise()).

    Scenario Parameters
    -------------------
    gamma : float
        Heat capacity ratio (1.4 for ideal gas).
    """
    NUM_VARIABLES = 3
    _VAR_RANGE = {
        "rho": 9.076252529801689,
        "rho_u": 962.4235162222517,
        "rho_e": 103620.30472859967,
    }

    # Initialization ----------------------------------------------------------
    def __init__(self, nx=200, nt=1000, L=2, tf=1e-2, gamma=1.4):
        """Initialize the domain and set variables for storing simulation data.

        Parameters
        ----------
        nx : int
            Number of partitions in the spatial domain, so that the total
            number of degrees of freedom is 3(nx + 1 - 1) = 3nx (periodicity).
        nt : int
            Number of partitions in the temporal domain.
        L : float
            Length of the spatial domain.
        tf : float
            Final simulation time.
        gamma : float
            Heat capacity ratio (1.4 for ideal gas).
        """
        self.gamma = gamma

        # Spatial domain
        self.x = np.linspace(0, L, nx+1)[:-1]           # Domain
        assert self._L == L                             # Length
        assert self._dx == L/nx                         # Resolution
        assert self.n == nx                             # Size
        assert self.x[-1] != L                          # Don't double count
        self._nodes = np.array([0, L/3, 2*L/3, L])      # Interpolation nodes

        # Temporal domain
        self.t = np.linspace(0, tf, nt+1)               # Temporal domain
        assert self._tf == tf                           # Length
        assert self._dt == round(tf/nt, 16)             # Resolution
        assert self.k == nt + 1                         # Size

        # Nondimensionalization constants.
        self.NONDIMENSIONALIZERS = {"u": 100, "rho": 10, "t": L/100}

    def _clear(self):
        """Reset the recorded state snapshots."""
        self.snapshots = []
        self.noise = []

    # Properties --------------------------------------------------------------
    @property
    def x(self):
        """Spatial domain."""
        return self.__x

    @x.setter
    def x(self, xx):
        """Reset the spatial domain, erasing all snapshot data."""
        self.__x = xx
        self.n = xx.size
        self._dx = xx[1] - xx[0]
        self._L = xx[-1] + self._dx
        self._clear()

    @property
    def t(self):
        """Temporal domain."""
        return self.__t

    @t.setter
    def t(self, tt):
        """Reset the temporal domain."""
        self.__t = tt
        self.k = tt.size
        self._dt = round(tt[1] - tt[0], 14)
        self._tf = tt[-1]
        self._clear()

    @property
    def _scalers(self):
        """Nondimensionalization scaling factors for u, p, and 1/rho:
        dimensional_variable / scaling_factor = nondimensional_variable.
        """
        _u = self.NONDIMENSIONALIZERS["u"]
        _rho = self.NONDIMENSIONALIZERS["rho"]
        return np.array([_u, _rho*_u**2, 1/_rho])

    def __len__(self):
        """Length: number of datasets."""
        return len(self.snapshots)

    def __getitem__(self, key):
        """Indexing: get a view of a subset of the saved data (NO COPIES)."""
        if isinstance(key, int):
            key = slice(key, key+1)
        elif not isinstance(key, slice):
            raise IndexError("key must be int or slice")
        if self.snapshots is None:
            raise IndexError("no data to select")

        newsolver = self.__class__(self.n, self.k-1,
                                   self._L, self._tf, self.gamma)
        newsolver.snapshots = self.snapshots[key]
        newsolver.noise = self.noise[key]
        return newsolver

    def extend_time(self, factor):
        """Extend / shorten the time domain, maintaining the step size.
        WARNING: This deletes all stored snapshot/derivative data!
        """
        self.t = _extend_time(self.t, factor)

    # Variable transformations ------------------------------------------------
    def lift(self, state):
        """LIFT from the conservative variables to the learning variables,
        [rho, rho*u, rho*e] -> [u, p, 1/rho].
        """
        rho, rho_u, rho_e = np.split(state, self.NUM_VARIABLES)

        u = rho_u / rho
        p = (self.gamma - 1)*(rho_e - 0.5*rho*u**2)
        zeta = 1 / rho

        return np.concatenate((u, p, zeta))

    def unlift(self, upzeta):
        """UNLIFT from the learning variables to the conservative variables,
        [u, p, 1/rho] -> [rho, rho*u, rho*e].
        """
        u, p, zeta = np.split(upzeta, self.NUM_VARIABLES)

        rho = 1/zeta
        rho_u = rho*u
        rho_e = p/(self.gamma - 1) + 0.5*rho*u**2

        return np.concatenate((rho, rho_u, rho_e))

    def nondimensionalize(self, upzeta):
        """Nondimensionalize the learning variables.

        Parameters
        ----------
        upzeta : (3n,...) ndarray
            Dimensional velocity, pressure, and 1/density in a single array.
            Each column (if 2D) corresponds to a single time step.

        Returns
        -------
        upzeta_nondim : (3n,...) ndarray
            Nondimensionalized velocity, pressure, and 1/density.
        """
        variables = np.split(upzeta, self.NUM_VARIABLES)
        return np.concatenate([v/s for v, s in zip(variables, self._scalers)])

    def redimensionalize(self, upzeta_nondim):
        """Unscale each learning variable.

        Parameters
        ----------
        upzeta_nondim : (3n,...) ndarray
            Nondimensional velocity, pressure, and 1/density.
            Each column (if 2D) corresponds to a single time step.

        Returns
        -------
        upzeta : (3n,...) ndarray
            Dimensionalized velocity, pressure, and 1/density.
        """
        variables = np.split(upzeta_nondim, self.NUM_VARIABLES)
        return np.concatenate([v*s for v, s in zip(variables, self._scalers)])

    def apply_noise(self, level):
        """Add Gaussian noise to the conservative variables.

        Parameters
        ----------
        level : float
            Scaling for the noise level of each variable, e.g., .05 for 5%.

        Returns
        -------
        snapshots_noised : (s,3n,...) ndarray
            Noisy velocity, pressure, and 1/density
        """
        if level == 0:
            return self.snapshots
        return [self.lift(self.unlift(snaps) + level*noise)
                for snaps, noise in zip(self.snapshots, self.noise)]

    # Initial and boundary conditions -----------------------------------------
    def initial_conditions(self, init_params, plot=False):
        """Generate initial conditions by evaluating periodic cubic splines for
        density rho and velocity u

        Parameters
        ----------
        init_params : (6,) ndarray
            Degrees of freedom for the initial conditions, three interpolation
            values for the density and three for the velocity (in that order).
        plot : bool
            If True, display the initial conditions over the spatial domain.

        Returns
        -------
        init : (3n,) ndarray
            Initial conditions in the LEARNING VARIABLES, [u, p, 1/rho].
        """
        # Unpack initial condition parameters.
        rho0s, u0s = np.split(np.array(init_params), 2)
        u0s = np.concatenate((u0s, [u0s[0]]))               # Make periodic.
        rho0s = np.concatenate((rho0s, [rho0s[0]]))         # Make periodic.

        # Initial condition for velocity.
        u_spline = interp.CubicSpline(self._nodes, u0s, bc_type="periodic")
        u = u_spline(self.x)

        # Initial condition for pressure.
        p = 1e5 * np.ones_like(u)

        # Initial condition for density.
        rho_spline = interp.CubicSpline(self._nodes, rho0s, bc_type="periodic")
        rho = rho_spline(self.x)

        # Group the initial conditions together and plot if desired.
        init = np.concatenate((u, p, 1/rho))
        if plot:
            fig, axes = self.plot_space(init)
            axes[0].set_title("Initial conditions")
            axes[0].plot(self._nodes, u0s, 'k*', mew=0)
            axes[2].plot(self._nodes, rho0s, 'k*', mew=0)
            plt.show()

        return init

    # High-fidelity solving ---------------------------------------------------
    def full_order_solve(self, init):
        """Solve the high-fidelity system for the given initial conditions.
        The solver operators on the conservative variables but saves results
        in the learning variables.

        Parameters
        ----------
        init : (3n,) ndarray
            Initial conditions in the LEARNING VARIABLES, [u, p, 1/rho].

        Returns
        -------
        snapshots : (3n,k) ndarray
            Solution to the PDE over the discretized space-time domain,
            in the LEARNING VARIABLES [u, p, 1/rho].
        """
        # Allocate space for solution snapshots.
        snapshots = np.zeros((self.NUM_VARIABLES*self.n, self.k), dtype=float)
        snapshots[:,0] = init
        dx, dt = self._dx, self._dt

        # Extract initial learning and conservative variables.
        u, p, zeta = np.split(init, self.NUM_VARIABLES)
        rho, rho_u, rho_e = np.split(self.unlift(init), self.NUM_VARIABLES)

        def ddx(v):
            """First-order spatial derivative via the (periodic) first-order
            backward finite difference formula (y[k] - y[k-1]) / δx.
            """
            return (v - np.roll(v, 1, axis=0)) / dx

        for i in range(1, self.k):
            # Integrate conservative variables with one forward Euler step.
            rho -= dt*ddx(rho*u)
            rho_u -= dt*ddx(rho*u**2 + p)
            rho_e -= dt*ddx((rho_e + p)*u)

            # Update velocity and pressure from conservative variables.
            snapshots[:,i] = self.lift(np.concatenate((rho, rho_u, rho_e)))
            u, p, zeta = np.split(snapshots[:,i], self.NUM_VARIABLES)

        return snapshots

    def add_snapshot_set(self, init, plot_init=False):
        """Get high-fidelity snapshots for the given initial condition
        parameters.

        Parameters
        ----------
        init : (3n,) ndarray OR (6,) ndarray OR None
            Initial conditions for the full-order solve. Options:
            * (3n,) ndarray: the initial conditions in the learning variables.
            * (6,) ndarray: Degrees of freedom for the initial conditions, three
                interpolation values for the density and three for the velocity.
            * None: Use random initial conditions.
        """
        # Get initial conditions.
        if init is None:
            init = self.random_initial_conditions(plot=plot_init)
        elif len(init) == 6:
            init = self.initial_conditions(init, plot=plot_init)

        # Run (and time) the full-order model
        with utils.timed_block("High-fidelity solve"):
            snaps = self.full_order_solve(init)

        # Generate noise for the full-order data.
        noise = np.vstack([np.random.normal(loc=0,
                                            scale=self._VAR_RANGE[var],
                                            size=(self.n, self.k))
                           for var in ["rho", "rho_u", "rho_e"]])

        # Record results.
        self.snapshots.append(snaps)
        self.noise.append(noise)

    # Visualization -----------------------------------------------------------
    def _format_spatial_subplots(self, fig, axes):
        """Put labels on subplots for variables in space."""
        axes[0].set_ylabel(r"Velocity $u(x)$", fontsize="x-large")
        axes[1].set_ylabel(r"Pressure $p(x)$", fontsize="x-large")
        axes[2].set_ylabel(r"Density $\rho(x)$", fontsize="x-large")
        fig.align_ylabels(axes)

        axes[2].set_xlabel(r"$x\in[0,L)$", fontsize="x-large")
        axes[2].set_xlim(self.x[0], self.x[-1])

    def _format_temporal_subplots(self, fig, axes):
        """Put labels on subplots for variables in time."""
        axes[0].set_ylabel(r"Velocity $u(x)$", fontsize="x-large")
        axes[1].set_ylabel(r"Pressure $p(x)$", fontsize="x-large")
        axes[2].set_ylabel(r"Density $\rho(x)$", fontsize="x-large")
        fig.align_ylabels(axes)

        axes[2].set_xlabel(r"$t\in[t_{0},t_{f}]$", fontsize="x-large")
        axes[2].set_xlim(self.t[0], self.t[-1])

    def plot_space(self, upzeta):
        """Plot velocity, pressure, and density over space at a fixed point
        in time.

        Parameters
        ----------
        upzeta: (3n,) ndarray
            velocity, pressure, and 1/density in a single array.

        Returns
        -------
        Figure handle, Axes handles
        """
        if isinstance(upzeta, int):
            upzeta = self.snapshots[upzeta]
        u, p, zeta = np.split(upzeta, self.NUM_VARIABLES)

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6,6))
        axes[0].plot(self.x, u)
        axes[1].plot(self.x, p)
        axes[2].plot(self.x, 1/zeta)
        self._format_spatial_subplots(fig, axes)

        return fig, axes

    def plot_time(self, u_p_or_zeta):
        """Plot velocity, pressure, or density in time at a fixed point in
        space.

        Parameters
        ----------
        upzeta: (3k,) ndarray
            velocity, pressure, and 1/density in a single array.

        Returns
        -------
        Figure handle, Axes handles
        """
        fig, ax = plt.subplots(1, 1, figsize=(6,2))
        ax.plot(self.t, u_p_or_zeta)
        ax.set_xlabel(r"$t\in[t_{0},t_{f}]$")
        ax.set_xlim(self.t[0], self.t[-1])
        return fig, ax

    def plot_traces(self, upzeta, nlocs=20, cmap=None):
        """Plot traces in time at `nlocs` locations."""
        if isinstance(upzeta, int):
            upzeta = self.snapshots[upzeta]

        xlocs = np.linspace(0, self.n, nlocs+1, dtype=np.int)[:-1]
        xlocs += xlocs[1]//2
        if cmap is None:
            cmap = plt.cm.twilight
        colors = cmap(np.linspace(0, 1, nlocs+1)[:-1])
        u, p, zeta = np.split(upzeta, self.NUM_VARIABLES)

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12,6))
        for j, c in zip(xlocs, colors):
            axes[0].plot(self.t, u[j], color=c, lw=1)
            axes[1].plot(self.t, p[j], color=c, lw=1)
            axes[2].plot(self.t, 1/zeta[j], color=c, lw=1)
        self._format_temporal_subplots(fig, axes)

        # Colorbar.
        lsc = cmap(np.linspace(0, 1, 400))
        scale = mplcolors.Normalize(vmin=0, vmax=1)
        lscmap = mplcolors.LinearSegmentedColormap.from_list("euler",
                                                             lsc, N=nlocs)
        mappable = plt.cm.ScalarMappable(norm=scale, cmap=lscmap)
        cbar = fig.colorbar(mappable, ax=axes, pad=0.015)
        cbar.set_ticks(self.x[xlocs] / self._L)
        cbar.set_ticklabels([f"{x:.2f}" for x in self.x[xlocs]])
        cbar.set_label(r"spatial coordinate $x$", fontsize="x-large")

        return fig, axes

    def plot_spacetime(self, upzeta):
        """Plot learning variables over space-time.

        Parameters
        ----------
        upzeta : (3n,k) ndarray
            The data to plot (learning variables).
        """
        # Process the input.
        if isinstance(upzeta, int):
            upzeta = self.snapshots[upzeta]
        if upzeta.ndim != 2:
            raise ValueError("arg must be two-dimensional")

        u, p, zeta = np.split(upzeta, self.NUM_VARIABLES)
        rho = 1/zeta
        X,T = np.meshgrid(self.x, self.t, indexing="ij")

        # Plot variables in space and time.
        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6,6))
        for v, ax in zip([u, p, rho], axes):
            cdata = ax.pcolormesh(X, T, v, shading="nearest", cmap="viridis")
            fig.colorbar(cdata, ax=ax, extend="both")
            ax.set_ylabel(r"$t\in[t_{0},t_{f}]$")

        axes[-1].set_xlabel(r"$x\in[0,L)$")
        axes[0].set_title("Velocity")
        axes[1].set_title("Pressure")
        axes[2].set_title("Density")

        return fig, axes

    def animate(self, profile, skip=20):
        """Animate a single evolution profile in time.

        Parameters
        ----------
        profile : (3n,k) ndarray
            In lifted variables...
        skip : int
            Animate every `skip` snapshots, so the total number of
            frames is `k//skip`
        """
        # Process the input.
        if isinstance(profile, int):
            profile = self.snapshots[profile]
        profile = np.array(profile)
        if profile.ndim != 2:
            raise ValueError("two-dimensional data required for animation")
        data = np.split(profile, 3, axis=0)

        # Initialize the figure and subplots.
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6,6), dpi=150)
        lines = [ax.plot([], [])[0] for ax in axes]

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(index):
            for ax, line, var in zip(axes, lines, data):
                line.set_data(self.x, var[:,index*skip])
            axes[0].set_title(fr"$t = t_{{{index*skip}}}$")
            return lines

        for ax, var in zip(axes, data):
            ax.set_ylim(var.min()*.95, var.max()*1.05)
        self._format_spatial_subplots(fig, axes)
        axes[0].set_title(r"$t = t_{0}$")

        a = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=profile.shape[1]//skip,
                                    interval=30, blit=True)
        plt.close(fig)
        return HTML(a.to_jshtml())

    # Data I/O ----------------------------------------------------------------
    @classmethod
    def load(cls, loadfile):
        """Load data from an HDF5 file.

        Should load domain data, snapshot data, etc.
        """
        with h5py.File(loadfile, 'r') as hf:

            # Domain parameters.
            if "domain" not in hf:
                raise ValueError("invalid save format (domain/ not found)")
            nx = hf["domain/nx"][0]
            nt = hf["domain/nt"][0]
            L = hf["domain/L"][0]
            tf = hf["domain/tf"][0]
            gamma = hf["domain/gamma"][0]
            solver = cls(nx, nt, L, tf, gamma)

            # Snapshot data.
            if "snapshots" in hf:
                solver.snapshots = [d for d in hf["snapshots/data"]]
                solver.noise = [n for n in hf["snapshots/noise"]]

        return solver

    def save(self, savefile, overwrite=False):
        """Save current state to an HDF5 file.

        Should save domain data, snapshot data, etc.

        Parameters
        ----------
        savefile : str
            The file to save to. If it does not end with '.h5', the extension
            will be tacked on to the end.

        overwrite : bool
            If True and the specified file already exists, overwrite the file.
            If False and the specified file already exists, raise an error.
        """
        # Make sure the file is saved in HDF5 format.
        if not savefile.endswith(".h5"):
            savefile += ".h5"

        # Do not overwrite existing file unless specified.
        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(savefile)

        # Create (or overwrite) the save file.
        with h5py.File(savefile, 'w') as hf:
            # Domain parameters.
            hf.create_dataset("domain/nx", data=[self.n])
            hf.create_dataset("domain/nt", data=[self.k-1])
            hf.create_dataset("domain/L", data=[self._L])
            hf.create_dataset("domain/tf", data=[self._tf])
            hf.create_dataset("domain/gamma", data=[self.gamma])

            # Snapshot data.
            if len(self.snapshots) > 0:
                hf.create_dataset("snapshots/data", data=self.snapshots)
                hf.create_dataset("snapshots/noise", data=self.noise)


# ROM utilities / classes =====================================================

def ddt_localpoly(n, k, dt, sparsify=False):
    """Construct a derivative estimation matrix via local polynomial fitting.

    This strategy is effective for estimating derivatives of noisy data without
    explicitly performing polynomial regression.

    Parameters
    ----------
    n : int
        Number of points at which the function to differentiate is measured.
    k : int
        Number of symmetric difference quotient terms in the approximation.
    dt : float
        Time step.

    Returns
    -------
    W : (n,n) ndarray
        Matrix such that if `u` is an (n,) ndarray of (noisy) function values
        with spacing `dt`, then `W @ u` approximates `du/dt`.

    References
    ----------
    [1] K. De Brabanter, J. De Brabanter, B. De Moor, and I. Gijbels,
        Derivative Estimation with Local Polynomial Fitting,
        Journal of Machine Learning Research, 14 (2013), pp 281-301.
    """
    W = sparse.coo_matrix((n,n), dtype=float) if sparsify else np.zeros((n,n))
    j = np.arange(1, k+1)
    w = 6*j**2 / (k*(k + 1)*(2*k + 1))
    dts = 2*j*dt

    # Fill interior rows.
    coeffs = w/dts
    row = np.concatenate([-coeffs[::-1], [0], coeffs])
    for i in range(k, n-k):
        W[i, i-k:i+k+1] = row

    # Fill boundary rows.
    for i in range(1, k):
        wi = w[:i] / w[:i].sum()
        coeffs = wi/dts[:i]
        row = np.concatenate([-coeffs[::-1], [0], coeffs])
        wlen = 2*i+1
        W[i, :wlen] = row
        W[n-i-1, -wlen:] = row

    return W.tocsc() if sparsify else W


def _kron_indices(r):
    """Construct masks for compact quadratic Kronecker."""
    r2_mask = np.zeros((r*(r+1)//2, 2), dtype=np.int)
    r2_count = 0
    for i in range(r):
        for j in range(i+1):
            r2_mask[r2_count,:] = (i,j)
            r2_count += 1
    return r2_mask


def rom_predict(rom, u0_, t):
    """Integrate a reduced-order model from given projected initial conditions
    using the Runge-Kutta 45 (adaptive) timestepping method.
    """
    if not hasattr(rom, "_r2"):
        rom._r2 = _kron_indices(rom.r)

    def _fun(t, u_):
        return rom.H_ @ np.prod(u_[rom._r2], axis=1)

    sol_ = sin.solve_ivp(_fun, [t[0], t[-1]], u0_, t_eval=t, method="RK45")
    if not sol_.success:
        np.warnings.warn(sol_.message, sin.IntegrationWarning)
    return sol_.y


class EulerROMSolver(EulerSolver):
    """Bundles a high-fidelity solver, data management, plotting tools, and
    ROM learning for 1D lifted Euler equations with periodic boundary
    conditions and random initial conditions.

    Attributes
    ----------
    snapshots : (s,3n,k) ndarray
        Snapshots, organized as s sets of (3n,k) ndarrays, one for each
        initial condition. Variables stored are [u, p, 1/rho].

    Scenario Parameters
    -------------------
    gamma : float
        Heat capacity ratio (1.4 for ideal gas).
    """
    _modelform = "H"

    # Reduced-order model construction ----------------------------------------
    def train_rom(self, r, reg=None, noise_level=0, margin=1.1, ktrain=None):
        """Use the stored snapshot data to compute an appropriate basis and
        train a ROM using Operator Inference.

        Parameters
        ----------
        r : int
            Number of POD basis vectors to use (size of the ROM).
        reg : float or None
            * float: Regularization hyperparameter λ.
            * None: do a gridsearch, then a 1D optimization to choose λ.
        noise_level : float
            Scaling for the noise level of each variable.
        margin : float
            Factor by which the integrated POD modes may deviate from training.
        ktrain : int or None
            Number of snapshots from each data set to use in training.
            If None, use all snapshots.
        """
        if self.snapshots is None:
            raise ValueError("no simulation data with which to train ROM")

        with utils.timed_block("preprocessing snapshots"):
            Us = self.snapshots
            if noise_level > 0:
                Us = self.apply_noise(noise_level)
            Us = [self.nondimensionalize(U[:,:ktrain]) for U in Us]

        with utils.timed_block("computing POD basis"):
            Vr = la.svd(np.hstack(Us), full_matrices=False)[0][:,:r]
            _r2 = _kron_indices(r)

        # Project the training data and calculate the derivative.
        with utils.timed_block(f"projecting training data (r = {r})"):
            Us_ = [Vr.T @ U for U in Us]
            if noise_level > 0:
                k = Us_[0].shape[1]
                W = ddt_localpoly(k, k//30, self._dt)
                Udots_ = [U_ @ W.T for U_ in Us_]
            # elif noise_level < 0:
            #     Udots_ = [Vr.T @ self._noisy_derivatives(Us, -noise_level)]
            else:
                Udots_ = [opinf.pre.xdot_uniform(U_, self._dt, 2)
                          for U_ in Us_]
            U_, Udot_ = np.hstack(Us_), np.hstack(Udots_)

        # Instantiate the ROM.
        rom = opinf.InferredContinuousROM(self._modelform)
        rom._training_states_ = U_
        rom._r2 = _r2

        # Single ROM solve, no regularization optimization.
        if reg is not None:
            with utils.timed_block(f"computing single ROM with λ={reg:5e}"):
                rom.reg = reg
                return rom.fit(Vr, U_, Udot_, P=reg)

        # Several ROM solves, optimizing the regularization.
        _MAXFUN = 1e12
        _BOUND = margin*np.abs(U_).max()
        with utils.timed_block("constructing OpInf least-squares solver"):
            rom._construct_solver(None, U_, Udot_, None, 1)

        def is_bounded(U):
            if U.shape[1] < self.k:
                print(f"ROM unstable after {U.shape[1]} steps", end="; ")
                return False
            elif np.abs(U).max() > _BOUND:
                print("bound exceeded...", end='')
                return False
            return True

        def rom_training_error(log10_λ):
            """Return the training error resulting from the regularization
            parameter λ = 10^log10_λ. If the resulting model violates the
            POD bound, return "infinity".
            """
            λ = 10**log10_λ
            error = 0

            with utils.timed_block(f"Testing ROM with λ={λ:e}"):
                rom._evaluate_solver(λ)

                # Test the ROM on each of the training initial conditions.
                for j,U__ in enumerate(Us_):
                    u0_ = U__[:,0]
                    try:
                        with np.warnings.catch_warnings():
                            np.warnings.simplefilter("error")
                            U_rom = rom_predict(rom, u0_, self.t)
                    except Exception as e:
                        print(f"ROM unstable ({type(e).__name__})", end="; ")
                        return _MAXFUN
                    else:
                        if not is_bounded(U_rom):
                            return _MAXFUN
                        error += opinf.post.Lp_error(U__,
                                                     U_rom[:,:ktrain],
                                                     self.t[:ktrain])[1]
                return error / len(Us_)

        # Evaluate rom_training_error() over a coarse logarithmic grid.
        print("starting regularization grid search")
        log10_grid = np.linspace(-16, 4, 81)
        errs = [rom_training_error(λ) for λ in log10_grid]
        windex = np.argmin(errs)
        λ = 10**log10_grid[windex]
        print(f"grid search winner: {λ:e}")
        if windex == 0 or windex == log10_grid.size - 1:
            print("WARNING: grid search bounds should be extended")
            rom._evaluate_solver(λ)
            rom.reg = λ
            rom.Vr = Vr
            return rom

        # Run the hyperparameter optimization and extract the best result.
        print("starting regularization optimization-based search")
        opt_result = opt.minimize_scalar(rom_training_error,
                                         bracket=log10_grid[windex-1:windex+2],
                                         method="brent")
        if opt_result.success and opt_result.fun != _MAXFUN:
            λ = 10 ** opt_result.x
            print(f"optimization-based search winner: {λ:e}")
            rom._evaluate_solver(λ)
            rom.reg = λ
            rom.Vr = Vr
            return rom
        else:
            print("regularization search optimization FAILED")

    def predict(self, rom, y0):
        """Integrate a reduced-order model from given initial conditions
        using the Runge-Kutta 45 (adaptive) timestepping method.
        """
        u0_ = rom.Vr.T @ self.nondimensionalize(y0)
        U_ = rom_predict(rom, u0_, self.t)
        return self.redimensionalize(rom.Vr @ U_)
