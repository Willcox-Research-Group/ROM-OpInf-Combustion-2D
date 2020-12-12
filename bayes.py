# bayes.py
"""Bayesian interpretation of Operator Inference for this problem.
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import rom_operator_inference as roi

import config
import utils
import data_processing as dproc
import step4_plot as step4


class _BaseOpInfPosterior:

    def _sample_operator_matrix(self):
        """Sample an operator matrix 'O' from the posterior distribution."""
        raise NotImplementedError("method must be implemented by child class")

    def rvs(self):
        """Do a single sample of the posterior distribution.

        Returns
        -------
        rom : roi.InferredContinuousROM
            A trained reduced-order model, representing a posterior draw.
        """
        O = self._sample_operator_matrix()
        c_, A_, H_, B_ = np.split(O, self._indices, axis=1)
        rom = roi.InferredContinuousROM(self._modelform)
        return rom._set_operators(None, c_=c_.flatten(), A_=A_, H_=H_, B_=B_)

    def predict(self, q0, t):
        """Draw a ROM from the posterior and simulate it from q0 over t."""
        return self.rvs().predict(q0, t, config.U, method="RK45")


class OpInfPosteriorUniformCov(_BaseOpInfPosterior):
    """Convenience class for sampling the posterior distribution, which
    is a different multivariate normal for each row of the operator matrix.

    This class is for the case where the intial guess for λ is a single
    number (same penalization for each operator entry), in which case
    the posterior distributions are N(µi, σi^2 Σ) (same Σ for each i).
    """
    def __init__(self, means, sigmas, Sigma, modelform="cAHB"):
        """Construct the multivariate normal random variables.

        Parameters
        ----------

        means (r,d) ndarray
            Mean values for each of the operator entries, i.e., E[O].

        """
        assert len(means) == len(sigmas)
        r = len(means)
        d = 1 + r + r*(r+1)//2 + 1
        assert means.shape == (r, d)
        self._d = d

        # Initialize the random variables for each operator row.
        self.means = means
        self.sigmas = sigmas
        self.Sigma = Sigma
        self._cho = np.linalg.cholesky(Sigma)

        # Set other convenience attributes.
        self._modelform = modelform
        if self._modelform == "cAHB":
            self._indices = np.cumsum([1, r, r*(r+1)//2])
        else:
            raise NotImplementedError(f"modelform='{self._modelform}'")

    def _sample_operator_matrix(self):
        """Sample an operator matrix from the posterior distribution."""
        rows = [µ + (σ*self._cho) @ np.random.standard_normal(self._d)
                                for µ, σ in zip(self.means, self.sigmas)]
        return np.vstack(rows)


def _data_matrix(trainsize, r):
    """Get the data matrix D for the given training size and
    ROM dimension.

    Returns
    -------
    D : (d(r,m),k) ndarray
        Data matrix for Operator Inference.

    R : (r,k) ndarray
        Right-hand side matrix for Operator Inference (time derivatives).
    """
    Q_, Qdot_, t, = utils.load_projected_data(trainsize, r)
    U = config.U(t).reshape((1,-1))
    D = roi.InferredContinuousROM("cAHB")._construct_data_matrix(Q_, U)
    return D, Qdot_


def _operator_matrix(trainsize, r, reg):
    """Extract the block matrix of learned operators O from the trained ROM.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROM. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    r : int
        The dimension of the ROM. This is also the number of retained POD modes
        (left singular vectors) used to project the training data.

    reg : float
        The regularization factor used in the Operator Inference least-squares
        problem for training the ROM.

    Returns
    -------
    rom : roi.InferredContinuousROM
        The trained reduced-order model.

    O : (r,d) ndarray
        Operator matrix O = [c | A | H | B].
    """
    if np.isscalar(reg):
        reg = [reg, reg]
    rom = utils.load_rom(trainsize, r, reg)
    return rom, np.column_stack((rom.c_, rom.A_, rom.H_, rom.B_))


def construct_posterior(trainsize, r, reg, case=3):
    """Construct the mean and covariance matrix for the posterior distribution,
    then create an object for sampling the posterior.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROM. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    r : int
        The dimension of the ROM. This is also the number of retained POD modes
        (left singular vectors) used to project the training data.

    reg : float
        The regularization factor used in the Operator Inference least-squares
        problem for training the ROM.

    case : int
        How to treat the prior regularzation / update.
        * 3: reg = a scalar, then Λ = λ I
        * 2: reg = (r,) ndarray, then Λi = λi I
        * 1: reg = (r,d) ndarray, then Λi = diag(λi1,...,λiM)
        Case c -> max(1, (c - 1)).

    Returns
    -------
    rom : roi.InferredContinuousROM
        The trained reduced-order model (mean of the posterior).

    post : OpInfPosterior
        Posterior distribution object with rvs() sampling method.
    """
    # Get the data matrix and solve the Operator Inference problem.
    Q_, R, t, = utils.load_projected_data(trainsize, r)
    U = config.U(t).reshape((1,-1))
    rom = roi.InferredContinuousROM("cAHB").fit(None, Q_, R, U, reg)
    rom.trainsize = trainsize
    D = rom._construct_data_matrix(Q_, U)
    O = rom.operator_matrix_

    # Check matrix shapes.
    d = 1 + r + r*(r+1)//2 + 1
    assert D.shape == (trainsize, d)
    assert R.shape == (r, trainsize)
    assert O.shape == (r, d)

    with utils.timed_block("Computing posterior parameters"):
        # Construct the data Grammian and the covariance matrix.
        DTD = D.T @ D
        DTD = (DTD + DTD.T) / 2     # Numerically symmetrize.
        λ = reg**2
        Λ = λ*np.eye(d)
        Σ = la.inv(DTD + Λ)

        # Numerically symmetrize / sparsify covariance for stability.
        Σ = (Σ + Σ.T) / 2
        Σ[np.abs(Σ) < 1e-16] = 0

        # Non-negative eigenvalues of data Grammian.
        gs = la.eigvalsh(DTD)
        gamma = np.sum(gs / (λ + gs))
        print("Gamma exact:", gamma)
        print("Gamma estimate:", d)

        # Get each covariance matrix.
        Onorms = np.sum(O**2, axis=1)       # ||o_i||^2.
        σ2s = (np.sum((D @ O.T - R.T)**2, axis=0) + λ*Onorms) / trainsize
        new_lambdas = gamma*σ2s/Onorms    # gamma
        print("Old log(λ):", np.log10(λ))
        print("New log(λ):", np.log10(new_lambdas))

    with utils.timed_block("Building posterior distribution"):
        return rom, OpInfPosteriorUniformCov(O, np.sqrt(σ2s), Σ, "cAHB")


def simulate_posterior(rom, post, ndraws=10, steps=None):
    """
    Parameters
    ----------
    rom : ...

    post : OpInfPosterior

    ndraws : int

    steps : int

    Returns
    -------
    q_rom_mean : (r,steps) ndarray
        TODO

    q_roms : list
        TODO

    scales : ndarray
        TODO
    """
    # Load the time domain and initial conditions.
    t = utils.load_time_domain(steps)
    Q_, _, _ = utils.load_projected_data(rom.trainsize, rom.r)
    q0 = Q_[:,0]

    # Simulate the mean ROM as a reference.
    with utils.timed_block(f"Simulating mean ROM"):
        q_rom_mean = rom.predict(q0, t, config.U, method="RK45")

    # Get ndraws simulation samples.
    q_roms = []
    i = 0
    while i < ndraws:
        with utils.timed_block(f"Simulating posterior draw ROM {i+1}"):
            q_rom = post.predict(q0, t)
            if q_rom.shape[1] == t.shape[0]:
                q_roms.append(q_rom)
                i += 1

    return q_rom_mean, q_roms


def plot_mode_uncertainty(trainsize, mean, draws, modes=8):
    steps = mean.shape[1]
    t = utils.load_time_domain(steps)

    with utils.timed_block("Calculating sample deviations"):
        offsets = [draw - mean for draw in draws]
        deviations = np.std(offsets, axis=0)

    nrows = (modes//2) + 1 if modes % 2 else modes//2
    fig, axes = plt.subplots(nrows, 2)
    for i, ax in zip(range(modes), axes.flat):
        ax.plot(t, mean[i,:], 'C0-', lw=1)
        # for draw in draws:
        #     ax.plot(t, draw[i,:], 'C0-', lw=.5, alpha=.2)
        ax.fill_between(t, mean[i,:]-deviations[i,:],
                           mean[i,:]+deviations[i,:], alpha=.5)
        ax.axvline(t[trainsize], color='k', lw=1)
        # ax.set_title(fr"POD mode ${i+1}$")
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
    fig.tight_layout()


def plot_pointtrace_uncertainty(trainsize, mean, draws, var="p"):
    """

    Parameters
    ----------

    Returns
    -------
    """
    if var not in ["p", "vx", "vy", "T"]:
        raise NotImplementedError(f"var='{var}'")

    # Get the indicies for each variable.
    elems = np.atleast_1d(config.MONITOR_LOCATIONS)
    nelems = elems.size
    nrows = (nelems // 2) + (1 if nelems % 2 != 0 else 0)
    elems = elems + config.ROM_VARIABLES.index(var)*config.DOF

    # Load the true pressure traces and the time domain.
    traces_gems, t = utils.load_gems_data(rows=elems)
    steps = mean.shape[1]
    t = t[:steps]
    traces_gems = traces_gems[:,:steps]

    # Load the basis rows corresponding to the pressure traces.
    V, scales = utils.load_basis(trainsize, mean.shape[0])
    Velems = V[elems]

    # Reconstruct and rescale the simulation results.
    with utils.timed_block("Reconstructing simulation results"):
        traces_rom_mean = dproc.unscale(Velems @ mean, scales, var)
        traces_rom_draws = [dproc.unscale(Velems @ draw, scales, var)
                            for draw in draws]

    with utils.timed_block("Calculating sample deviations"):
        offsets = [draw - traces_rom_mean for draw in traces_rom_draws]
        deviations = np.std(offsets, axis=0)

    fig, axes = plt.subplots(nrows, 2, figsize=(9,6), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(t, traces_gems[i,:], lw=1, **config.GEMS_STYLE)
        ax.plot(t, traces_rom_mean[i,:], 'C0-', lw=1,
                label=r"ROM ($\mu$)")
        # for draw in traces_rom_draws:
        #     ax.plot(t, draw[i,:], 'C0-', lw=.5, alpha=.25)
        ax.fill_between(t, traces_rom_mean[i,:]-deviations[i,:],
                           traces_rom_mean[i,:]+deviations[i,:], alpha=.5)
        ax.axvline(t[trainsize], color='k', lw=1)
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
        ax.set_title(f"Location ${i+1}$", fontsize=12)
        ax.locator_params(axis='y', nbins=2)
    for ax in axes[-1,:]:
        ax.set_xlabel("Time [s]", fontsize=12)
    for ax in axes[:,0]:
        ax.set_ylabel(config.VARLABELS[var], fontsize=12)

    # Single legend to the right of the subplots.
    fig.tight_layout(rect=[0, 0, .85, 1])
    leg = axes[0,0].legend(loc="center right", fontsize=14,
                           bbox_to_anchor=(1,.5),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(2)


def main(trainsize, r, reg, ndraws=10, steps=50000, modes=4):
    rom, post = construct_posterior(trainsize, r, reg)
    mean, draws = simulate_posterior(rom, post, ndraws, steps)
    plot_mode_uncertainty(trainsize, mean, draws, modes)
    utils.save_figure("bayes_first4modes.pdf")
    plot_pointtrace_uncertainty(trainsize, mean, draws, var="p")
    utils.save_figure("bayes_traces_pressure.pdf")
    plot_pointtrace_uncertainty(trainsize, mean, draws, var="T")
    utils.save_figure("bayes_traces_temperature.pdf")


if __name__ == "__main__":
    main(20000, 40, 36382, 100, 50000, 4)
