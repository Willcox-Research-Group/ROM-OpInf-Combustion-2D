# bayes.py
"""Bayesian Operator Inference for this problem."""

import logging
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import rom_operator_inference as roi

import config
import utils
import step3_train as step3


# Posterior samplers ==========================================================

class OpInfPosterior:
    """Convenience class for sampling the posterior distribution, which
    is a different multivariate normal for each row of the operator matrix.
    """
    def __init__(self, means, Sigmas, modelform="cAHB"):
        """Construct the multivariate normal random variables.

        Parameters
        ----------
        means : (r,d) ndarray
            Mean values for each of the operator entries, i.e., Mean(Ohat).
        Sigmas : (r,d,d) ndarray
            Covariance matrices for each row of the operator matrix.
            That is, Sigmas[i] = Covariance(Ohat[i])
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

    def _construct_rom(self, Ohat):
        """Construct the ROM from the operator matrix."""
        c_, A_, H_, B_ = np.split(Ohat, self._indices, axis=1)
        rom = roi.InferredContinuousROM(self._modelform)
        return rom.set_operators(None, c_=c_.flatten(), A_=A_, H_=H_, B_=B_)

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

    This class is only for the special case in which the intial guess for λ
    is a single number (same penalization for each operator entry), resulting
    in the posterior distributions N(µi, σi^2 Σ) (i.e., same Σ for each i).
    """
    def __init__(self, means, sigmas, Sigma, modelform="cAHB"):
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
        self._Sigma = Sigma
        self._cho = np.linalg.cholesky(Sigma)

    @property
    def covariances(self):
        """Covariance matrices for each row of the operator matrix."""
        return np.array([(σ**2)*self._Sigma for σ in self._sigmas])

    def _sample_operator_matrix(self):
        """Sample an operator matrix from the posterior distribution."""
        rows = [µ + (σ*self._cho) @ np.random.standard_normal(size=self._d)
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

    Returns
    -------
    post : OpInfPosterior
        Posterior distribution object with rvs() sampling method.

    reg_new : (r,) ndarray or (r,d) ndarray
        The Bayesian update for the regularization parameters.
        * case == 2 -> (r,), a new λ for each ROW of the operator matrix.
        * case == 1 -> (r,d), a new λ for each ENTRY of the operator matrix.
    """
    d = 1 + r + r*(r+1)//2 + 1
    if isinstance(reg, tuple) and len(reg) == 2:
        reg = step3.regularizer(r, reg[0], reg[1])

    # Get the data matrix and solve the Operator Inference problem,
    # using the initial guess for the regularization parameter(s).
    Q_, R, t, = utils.load_projected_data(trainsize, r)
    U = config.U(t).reshape((1,-1))
    rom = roi.InferredContinuousROM("cAHB").fit(None, Q_, R, U, reg)
    rom.trainsize = trainsize
    D = rom._assemble_data_matrix(Q_, U)
    Ohat = rom.O_

    # Check matrix shapes.
    assert D.shape == (trainsize, d)
    assert R.shape == (r, trainsize)
    assert Ohat.shape == (r, d)

    def symmetrize(S, sparsify=False):
        """Numerically symmetrize / sparsify (e.g., for covariance)."""
        S = (S + S.T) / 2
        if sparsify:
            S[np.abs(S) < 1e-16] = 0
        return S

    with utils.timed_block("Building posterior distribution"):
        # Precompute some quantities for posterior parameters.
        DTD = symmetrize(D.T @ D)
        Ohatnorm2s = np.sum(Ohat**2, axis=1)                # ||o_i||^2.
        residual2s = np.sum((D @ Ohat.T - R.T)**2, axis=0)  # ||Do_i - r_i||^2.

        # print("||o_i||^2:", Ohatnorm2s)
        # print(f"{Ohatnorm2s.mean()} ± {Ohatnorm2s.std()}")
        # print("||Do_i - r_i||^2:", residual2s)
        # print(f"{residual2s.mean()} ± {residual2s.std()}")
        # input("Press ENTER to continue")

        # Calculate posterior ROM distribution.
        if np.isscalar(reg):
            λ2 = reg**2
            Λ = λ2*np.eye(d)
            Σ = symmetrize(la.inv(DTD + Λ), sparsify=True)
            σ2s = (residual2s + λ2*Ohatnorm2s) / trainsize
            post = OpInfPosteriorUniformCov(Ohat, np.sqrt(σ2s), Σ, "cAHB")
            if case == 1:
                Σs = np.array([σ2i * Σ for σ2i in σ2s])  # = post.covariances
        else:
            if reg.shape == (d,):
                # TODO: this is Gamma_{i} = Gamma_{fixed},
                # which would mean DTD + Gamma_{i} is the same each time,
                # so we could speed this part up quite a bit.
                reg = np.tile(reg, (r,1))
            λ2 = np.array(reg)**2
            if λ2.shape == (r,):
                Id = np.eye(d)
                Λs = [λ2i*Id for λ2i in λ2]
                σ2s = (residual2s + λ2*Ohatnorm2s) / trainsize
            elif λ2.shape == (r,d):
                if case not in (-1, 1):
                    raise ValueError("2D reg only compatible with case=1")
                Λs = [np.diag(λ2i) for λ2i in λ2]
                σ2s = (residual2s + np.sum(λ2*Ohat**2, axis=1)) / trainsize
            else:
                raise ValueError("invalid shape(reg)")
            assert len(Λs) == len(σ2s) == r
            Σs = np.array([σ2i * symmetrize(la.inv(DTD + Λi), sparsify=True)
                           for σ2i, Λi in zip(σ2s, Λs)])
            post = None
            post = OpInfPosterior(Ohat, Σs, modelform="cAHB")

    if case == -1:
        return post

    with utils.timed_block("Calculating updated regularization parameters"):
        if case == 2:  # So λ2 is a scalar or (r,) ndarray
            gs = la.eigvalsh(DTD)   # Non-negative eigenvalues of data Grammian.
            if np.isscalar(reg):    # Scalar regularization parameter
                gamma = np.sum(gs / (λ2 + gs))
            else:
                gamma = np.sum(gs / (λ2.reshape((-1,1)) + gs), axis=1)
                assert len(gamma) == r
            # print("Gamma exact:", gamma)
            # print("Gamma estimate (d):", d)
            λ2_new = gamma*σ2s/Ohatnorm2s
        elif case == 1:
            # TODO: verify and fix. Note use of λ2_new, not λ_new.
            xi = np.zeros_like(Ohat)
            badmask = np.ones_like(Ohat, dtype=bool)
            pairs = []
            for i in range(Ohat.shape[0]):
                for j in range(Ohat.shape[1]):
                    pairs.append((Σs[i,j,j], Ohat[i,j]**2))
                    s = Σs[i,j,j] / Ohat[i,j]**2
                    if s < 1:
                        xi[i,j] = 1 - s + s**2 - s**3 + s**4 - s**5 + s**6
                        badmask[i,j] = False
            λ2_new = (xi / Ohat**2) * σ2s.reshape((-1,1))
            λ2_new[badmask] = .01
            assert λ2_new.shape == (r,d)

            pairs = np.array(pairs)
            bad = (pairs[:,0] > pairs[:,1])
            plt.plot(pairs[~bad,0], pairs[~bad,1], 'C0.', ms=2, alpha=.2)
            plt.plot(pairs[bad,0], pairs[bad,1], 'C3.', ms=2, alpha=.2)
            plt.show()

        else:
            raise ValueError(f"invalid case ({case})")

    return post, np.sqrt(λ2_new)


# Posterior simulation ========================================================
def simulate_posterior(trainsize, post, ndraws=10, steps=None):
    """
    Parameters
    ----------
    rom :
        TODO
    post : OpInfPosterior
        TODO
    ndraws : int
        TODO
    steps : int
        TODO

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
    with utils.timed_block("Simulating mean ROM"):
        q_rom_mean = post.mean_rom.predict(q0, t, config.U, method="RK45")

    # Get `ndraws` simulation samples.
    q_roms = []
    i, failures = 0, 0
    while i < ndraws:
        with utils.timed_block(f"Simulating posterior draw ROM {i+1:0>3d}"):
            q_rom = post.predict(q0, t)
            if q_rom.shape[1] == t.shape[0]:
                q_roms.append(q_rom)
                i += 1
            else:
                print("UNSTABLE...", end='')
                failures += 1
    if failures:
        logging.info(f"TOTAL FAILURES: {failures}")

    return q_rom_mean, q_roms
