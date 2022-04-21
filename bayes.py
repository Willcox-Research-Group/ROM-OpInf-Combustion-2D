# bayes.py
"""Bayesian Operator Inference."""

import logging
import numpy as np
import scipy.linalg as la

import rom_operator_inference as opinf

import utils


# Posterior samplers ==========================================================

class OpInfPosterior:
    """Convenience class for sampling the posterior distribution, which
    is a different multivariate normal for each row of the operator matrix.
    """
    def __init__(self, means, invSigmas, modelform="cAHB"):
        """Construct the multivariate normal random variables.

        Parameters
        ----------
        means : (r,d) ndarray
            Mean values for each of the operator entries, i.e., Mean(Ohat).
        invSigmas : (r,d,d) ndarray
            INVERSE covariance matrices for each row of the operator matrix.
            That is, MatrixInverse(invSigmas[i]) = Covariance(Ohat[i])
        modelform : str
            Structure of the ROMs to sample.
        """
        self._init_means(means, modelform)

        # Factor the covariances for each operator row.
        self.invcovariances = np.array(invSigmas)
        self._invchos = [la.cholesky(Σinv) for Σinv in invSigmas]
        self._piv = np.arange(self._d, dtype=np.int32)

    def _init_means(self, means, modelform):
        """Record means and initialize dimensions."""
        self.means = np.array(means)

        # Check and save dimensions.
        r = means.shape[0]
        self._m = 1 if "B" in modelform else 0
        self._r, self._d = r, opinf.lstsq.lstsq_size(modelform, r, self._m)
        assert self.means.shape == (self._r, self._d)

        self._modelform = modelform

    def _sample_operator_matrix(self):
        """Sample an operator matrix from the posterior distribution."""
        snrvs = np.random.standard_normal(self._d)
        # N(µ, Σ) -> µ + U^{-1} @ snrvs, U.T @ U = Σ^{-1}
        return np.vstack([µ + la.lu_solve((invcho, self._piv), snrvs)
                          for µ, invcho in zip(self.means, self._invchos)])

    def _construct_rom(self, Ohat):
        """Construct the ROM from the operator matrix."""
        rom = opinf.InferredContinuousROM(self._modelform)
        rom.r, rom.m = self._r, self._m
        rom._extract_operators(Ohat)
        return rom

    @property
    def mean_rom(self):
        """Get the mean OpInf ROM from the posterior."""
        return self._construct_rom(self.means)

    def rvs(self):
        """Do a single sample of the posterior OpInf ROM distribution.

        Returns
        -------
        rom : opinf.InferredContinuousROM
            A trained reduced-order model, representing a posterior draw.
        """
        return self._construct_rom(self._sample_operator_matrix())

    def predict(self, q0, t, input_func=None):
        """Draw a ROM from the posterior and simulate it from q0 over t."""
        return self.rvs().predict(q0, t, input_func, method="RK45")


class OpInfPosteriorUniformCov(OpInfPosterior):
    """Convenience class for sampling the posterior distribution, which
    is a different multivariate normal for each row of the operator matrix.

    This class is only for the special case in which the intial guess for λ
    is a single number (same penalization for each operator entry), resulting
    in the posterior distributions N(µi, σi^2 Σ) (i.e., same Σ for each i).
    This requires only one Cholesky factorization as opposed to r
    factorizations in the more general case.
    """
    def __init__(self, means, sigmas, invSigma, modelform="cAHB"):
        """Construct the multivariate normal random variables.

        Parameters
        ----------
        means : (r,d) ndarray
            Mean values for each of the operator entries, i.e., E[Ohat].
        sigmas : list of r floats or (r,) ndarray
            Scaling factors for each covariance matrix.
        Sigma : (d,d) ndarray
            Nonscaled covariance matrix for each posterior.
        """
        self._init_means(means, modelform)

        # Factor the covariances for each operator row.
        self._sigmas = sigmas
        self._invSigma = invSigma
        self._invcho = la.cholesky(invSigma)
        self._piv = np.arange(self._d, dtype=np.int32)

    @property
    def invcovariances(self):
        """Covariance matrices for each row of the operator matrix."""
        return np.array([self._invSigma/(σ**2) for σ in self._sigmas])

    def _sample_operator_matrix(self):
        """Sample an operator matrix from the posterior distribution."""
        snrvs = np.random.standard_normal(size=self._d)
        return np.vstack([µ + la.lu_solve(((self._invcho/σ), self._piv), snrvs)
                          for µ, σ in zip(self.means, self._sigmas)])


# Posterior construction ======================================================

def _symmetrize(S, sparsify=False, cutoff=1e-16):
    """Numerically symmetrize / sparsify (e.g., for covariance)."""
    S = (S + S.T) / 2
    if sparsify:
        S[np.abs(S) < cutoff] = 0
    return S


def construct_posterior(rom, reg, case=-1, gs=None):
    """Construct the mean and covariance matrix for the posterior distribution,
    then create an object for sampling the posterior.

    Parameters
    ----------
    rom : opinf.ContinuousROM
        *Trained* OpInf ROM object (meaning _construct_solver() was called).
    reg : float or (r,) ndarray or (r,d) ndarray
        The regularization parameter(s) used in the Operator Inference
        least-squares problem for training the ROM.
        * float: Λ = λi I
        * (r,) ndarray: Λi = λi I
        * (r,d) ndarray: Λi = diag(λi1,...,λid) (requires case=1)
    case : int
        How to treat the regularization update.
        * 2: learn a new λ for each ROW of the operator matrix.
        * 1: learn a new λ for each ENTRY of the operator matrix.
        * -1: do not update λ, just return `post`.
    gs : (d,) ndarray or None
        Eigenvalues of the data Grammian D.T @ D.
        If None, computed when needed.

    Returns
    -------
    post : OpInfPosterior
        Posterior distribution object with rvs() sampling method.
    reg_new : (r,) ndarray or (r,d) ndarray
        The Bayesian update for the regularization parameters.
        * case == 2 -> (r,), a new λ for each ROW of the operator matrix.
        * case == 1 -> (r,d), a new λ for each ENTRY of the operator matrix.
    """
    # Get the data, derivative, and operator matrices.
    D = rom.solver_.A
    Ohat = rom.solver_.predict(reg).T

    r, d = Ohat.shape
    trainsize = D.shape[0]
    assert D.shape[1] == d

    # Precompute some quantities for posterior parameters.
    DTD = _symmetrize(D.T @ D)
    σ2s = rom.solver_.residual(Ohat.T, reg) / trainsize

    # Calculate posterior ROM distribution.
    if np.isscalar(reg):
        λ2 = reg**2
        Λ = λ2*np.eye(d)
        Σinv = DTD + Λ
        Σinvs = np.array([Σinv / σ2i for σ2i in σ2s])
        post = OpInfPosteriorUniformCov(Ohat, np.sqrt(σ2s), Σinv,
                                        modelform=rom.modelform)
    else:
        if reg.shape == (d,):
            reg = np.tile(reg, (r,1))
        λ2 = np.array(reg)**2
        if λ2.shape == (r,):
            # One regularization for each ROW (output of case 2).
            Id = np.eye(d)
            Λs = [λ2i*Id for λ2i in λ2]
        elif λ2.shape == (r,d):
            # One regularization for each ENTRY (output of case 1).
            Λs = [np.diag(λ2i) for λ2i in λ2]
        else:
            raise ValueError("invalid shape(reg)")
        assert len(Λs) == len(σ2s) == r
        Σinvs = np.array([(DTD + Λi) / σ2i for σ2i, Λi in zip(σ2s, Λs)])
        post = OpInfPosterior(Ohat, Σinvs, modelform=rom.modelform)

    # Calculate the regularization update (or quit early and return post).
    if case == -1:
        return post
    elif case == 1:
        Σdiags = np.array([np.diag(la.inv(Σinv)) for Σinv in Σinvs])
        λ2_new = σ2s[:,None] / (Ohat**2 + Σdiags)
        assert λ2_new.shape == (r,d)
    elif case == 2:
        if gs is None:
            gs = la.eigvalsh(DTD)   # Non-negative eigenvalues of Grammian.
        if np.isscalar(reg):    # Scalar regularization parameter.
            gamma = np.sum(gs / (λ2 + gs))
        else:
            gamma = np.sum(gs / (λ2.reshape((-1,1)) + gs), axis=1)
            assert len(gamma) == r
        λ2_new = gamma * σ2s / np.sum(Ohat**2, axis=1)
        assert λ2_new.shape == (r,)
    elif case == 3:
        assert rom.modelform == "cAHB"
        Σdiags = np.array([np.diag(la.inv(Σinv)) for Σinv in Σinvs])
        denom = np.zeros_like(Ohat)
        for j in [0, -1]:
            denom[:,j] = Ohat[:,j]**2 + Σdiags[:,j]
        for s in [slice(1, 1 + r), slice(1 + r, 1 + r + (r*(r+1))//2)]:
            denom[:,s] = np.tile(np.mean(Ohat[:,s]**2 + Σdiags[:,s], axis=1),
                                 (s.stop - s.start, 1)).T
        λ2_new = σ2s[:,None] / denom
        return None, np.sqrt(λ2_new)
    else:
        raise ValueError(f"invalid case ({case})")

    return post, np.sqrt(λ2_new)


# Posterior simulation ========================================================
def simulate_posterior(post, q0, t, input_func=None, ndraws=10):
    """Simulate multiple posterior draws.

    Parameters
    ----------
    post : OpInfPosterior
        Posterior OpInf ROM object.
    q0 : (r,) ndarray
        Projected initial conditions for the ROM.
    t : (k,) ndarray
        Time domain over which to integrate ROM.
    input_func : func or None
        Function for inputs (if present).
    ndraws : int
        Number of simulation samples to draw.

    Returns
    -------
    q_rom_mean : (r,k) ndarray
        Results of integrating the mean ROM.
    q_roms : list(ndraws (r,k) ndarrays)
        Results of integrating ndraws ROMs from the posterior distribution.
   """
    # Simulate the mean ROM as a reference.
    with utils.timed_block("Simulating mean ROM"):
        q_rom_mean = post.mean_rom.predict(q0, t, input_func, method="RK45")

    # Try to get `ndraws` simulation samples.
    q_roms, failures = [], 0
    with utils.timed_block(f"Simulating {ndraws} posterior ROM draws"):
        for i in range(ndraws):
            q_rom = post.rvs().predict(q0, t, input_func, method="RK45")
            if q_rom.shape[1] == t.shape[0]:
                q_roms.append(q_rom)
            else:
                print("UNSTABLE...", end='')
                failures += 1
    if failures:
        message = f"UNSTABLE DRAWS: {failures}"
        print(message)
        logging.info(message)

    return q_rom_mean, q_roms

