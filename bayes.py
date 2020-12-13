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


# Operator Inference posterior samplers =======================================

class OpInfPosterior:
    """Convenience class for sampling the posterior distribution, which
    is a different multivariate normal for each row of the operator matrix.
    """
    def __init__(self, means, Sigmas, modelform="cAHB"):
        """Construct the multivariate normal random variables.

        Parameters
        ----------
        means: list of r (d,) ndarrays or (r,d) ndarray
            Mean values for each of the operator entries, i.e., Mean(O).

        Sigmas: list of r (d,d) ndarrays or (r,d,d) ndarray
            Covariance matrices for each row of the operator matrix.
            That is, Sigmas[i] = Covariance(O[i])

        modelform : str
            Structure of the ROMs to sample.
        """
        self._init_means(means, modelform)

        # Factor the covariances for each operator row.
        self.covariances = np.array(Sigmas)
        self._chos = [np.linalg.cholesky(Σ) for Σ in Sigmas]

    def _init_means(self, means, modelform, m=1):
        """Record means and initialize dimensions."""
        # Check and save dimensions.
        self.means = np.array(means)
        r = means.shape[0]
        self._r, self._d = r, roi.lstsq.lstsq_size(modelform, r, m)
        assert self.means.shape == (self._r, self._d)

        # Get split indices for the operator matrix.
        if modelform == "cAHB":
            self._indices = np.cumsum([1, r, r*(r+1)//2])
        else:
            raise NotImplementedError(f"modelform='{modelform}'")
        self._modelform = modelform

    def _sample_operator_matrix(self):
        """Sample an operator matrix from the posterior distribution."""
        rows = [µ + cho @ np.random.standard_normal(self._d)
                                for µ, cho in zip(self.means, self._chos)]
        return np.vstack(rows)

    def _construct_rom(self, O):
        """Construct the ROM from the operator matrix."""
        c_, A_, H_, B_ = np.split(O, self._indices, axis=1)
        rom = roi.InferredContinuousROM(self._modelform)
        return rom._set_operators(None, c_=c_.flatten(), A_=A_, H_=H_, B_=B_)

    @property
    def mean_rom(self):
        """Get the mean OpInf ROM from the posterior."""
        return self._construct_rom(self.means)

    def rvs(self):
        """Do a single sample of the posterior OpInf ROM distribution.

        Returns
        -------
        rom : roi.InferredContinuousROM
            A trained reduced-order model, representing a posterior draw.
        """
        return self._construct_rom(self._sample_operator_matrix())

    def predict(self, q0, t):
        """Draw a ROM from the posterior and simulate it from q0 over t."""
        return self.rvs().predict(q0, t, config.U, method="RK45")


class OpInfPosteriorUniformCov(OpInfPosterior):
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
        self._init_means(means, modelform)

        # Factor the covariances for each operator row.
        self._sigmas = sigmas
        self._Sigma = Sigma
        self._cho = np.linalg.cholesky(Sigma)

    @property
    def covariances(self):
        """Covariance matrices for each row of the operator matrix."""
        return np.array([(σ**2)*self._Sigma for σ in self._sigmas])

    def _sample_operator_matrix(self):
        """Sample an operator matrix from the posterior distribution."""
        rows = [µ + (σ*self._cho) @ np.random.standard_normal(self._d)
                                for µ, σ in zip(self.means, self._sigmas)]
        return np.vstack(rows)


# Posterior construction ======================================================

def construct_posterior(trainsize, r, reg, case=2):
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
        How to treat the regularization update.
        * 2: Λi = λi I (learn a λ for each row of O).
        * 1: Λi = diag(λi1,...,λiM) (learn a λ for each entry of O).

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
    D = rom._assemble_data_matrix(Q_, U)
    O = rom.operator_matrix_

    # Check matrix shapes.
    d = 1 + r + r*(r+1)//2 + 1
    assert D.shape == (trainsize, d)
    assert R.shape == (r, trainsize)
    assert O.shape == (r, d)

    def symmetrize(S, sparsify=False):
        """Numerically symmetrize / sparsify covariance for stability."""
        S = (S + S.T) / 2
        if sparsify:
            S[np.abs(S) < 1e-16] = 0
        return S

    with utils.timed_block("Building posterior distribution"):
        # Precompute some quantities for posterior parameters.
        DTD = symmetrize(D.T @ D)
        Onorms = np.sum(O**2, axis=1)                   # ||o_i||^2.
        residual2s = np.sum((D @ O.T - R.T)**2, axis=0) # ||Do_i - r_i||^2.

        # Calculate posterior ROM distribution.
        if np.isscalar(reg):
            λ = reg**2
            Λ = λ*np.eye(d)
            Σ = symmetrize(la.inv(DTD + Λ), sparsify=True)
            σ2s = (residual2s + np.sum(λ*O**2, axis=1)) / trainsize
            post = OpInfPosteriorUniformCov(O, np.sqrt(σ2s), Σ, "cAHB")
            if case == 1:
                Σs = np.array([σ2i * Σ for σ2i in σ2s])
        else:
            λ = np.array(reg)
            if λ.shape == (r,):
                I = np.eye(d)
                Λs = [λi*I for λi in λ]
            elif λ.shape == (r,d):
                if case != 1:
                    raise ValueError("2D reg only compatible with case=1")
                Λs = [np.diag(λi) for λi in λ]
            else:
                raise ValueError("invalid shape(reg)")
            σ2s = (residual2s + np.sum(λ*O**2, axis=1)) / trainsize
            Σs = np.array([σ2i * symmetrize(la.inv(DTD + Λi), sparsify=True)
                                                for σ2i, Λi in zip(σ2s, Λs)])
            post = OpInfPosterior(O, Σs, modelform="cAHB")

    with utils.timed_block("Calculating updated regularization parameters"):
        if case == 2:
            # Non-negative eigenvalues of data Grammian.
            gs = la.eigvalsh(DTD)
            gamma = np.sum(gs / (λ + gs)) # What if λ = [λ1, ..., λr]?
            # print("Gamma exact:", gamma)
            # print("Gamma estimate:", d)
            λ_new = gamma*σ2s/Onorms    # gamma
            print("\nOld log(λ):", np.log10(λ))
            print("New log(λ):", np.log10(λ_new), sep='\n')
        elif case == 1:
            xi = np.empty_like(O)
            for i in range(O.shape[0]):
                for j in range(O.shape[1]):
                    # print(Σs[i,j,j], O[i,j]**2)
                    s = Σs[i,j,j] / O[i,j]**2
                    xi[i,j] = 1 - s + s**2 - s**3 + s**4 - s**5 + s**6 - s**7
            print(xi.min(), xi.max(), xi.mean(), xi.std())
            λ_new = (xi / O**2) * σ2s.reshape((-1,1))
            λ_new = (1 / O**2) * σ2s.reshape((-1,1))
            import sys; sys.exit(1)
        else:
            raise ValueError(f"invalid case ({case})")

    return post, λ_new


def simulate_posterior(trainsize, post, ndraws=10, steps=None):
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
    q0 = utils.load_projected_data(trainsize, post._r)[0][:,0]

    # Simulate the mean ROM as a reference.
    with utils.timed_block(f"Simulating mean ROM"):
        q_rom_mean = post.mean_rom.predict(q0, t, config.U, method="RK45")

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


# Main routines ===============================================================

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
        ax.fill_between(t, mean[i,:] - 3*deviations[i,:],
                           mean[i,:] + 3*deviations[i,:], alpha=.5)
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
        ax.fill_between(t, traces_rom_mean[i,:] - 3*deviations[i,:],
                           traces_rom_mean[i,:] + 3*deviations[i,:], alpha=.5)
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
    post = construct_posterior(trainsize, r, reg, case=2)[0]
    mean, draws = simulate_posterior(trainsize, post, ndraws, steps)
    plot_mode_uncertainty(trainsize, mean, draws, modes)
    utils.save_figure("bayes_first4modes.pdf")
    plot_pointtrace_uncertainty(trainsize, mean, draws, var="p")
    utils.save_figure("bayes_traces_pressure.pdf")
    plot_pointtrace_uncertainty(trainsize, mean, draws, var="T")
    utils.save_figure("bayes_traces_temperature.pdf")


if __name__ == "__main__":
    main(20000, 40, 36382, 100, 50000, 4)
