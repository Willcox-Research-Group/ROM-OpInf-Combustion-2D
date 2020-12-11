# combustion_rom.py
"""Custom Operator Inference ROM class for the combustion problem,
with a separate basis for non-temperature variables and temperature,
with a quadratic model for the non-temperature equations and a
cubic model for the temperature equations.
"""
import numpy as np

import rom_operator_inference as roi
kron2c = roi.utils.kron2c
kron3c = roi.utils.kron3c


# Utilities ===================================================================

class MultiLstsqSolver(roi.lstsq._tikhonov._BaseSolver):
    """Encapsulate multiple least-squares solvers."""
    def fit(self, As, Bs, SolverClass):
        """Initialize and fit a solver for each pair of inputs."""
        # Check inputs.
        if len(As) != len(Bs):
            raise ValueError("inputs not aligned")
        if not issubclass(SolverClass, roi.lstsq._tikhonov._BaseSolver):
            raise TypeError("invalid SolverClass")

        # Fit each solver.
        self.solvers_ = [SolverClass(compute_extras=False,
                                     check_regularizer=False).fit(A,B)
                         for A,B in zip(As, Bs)]
        return self

    def predict(self, Ps):
        """Predict with each solver."""
        if len(Ps) != len(self.solvers_):
            raise ValueError("len(Ps) != # of separate lstsq problems")
        return [solver.predict(P) for solver, P in zip(self.solvers_, Ps)]


class CombustionROM(roi.InferredContinuousROM):
    """Operator Inference ROM with a specific structure:
    * cAHB (quadratic with input) for the first r1 modes
    * cAHGB (cubic with input) for the last r2 modes
    """
    def __init__(self, r1, r2, v):
        if v > r2:
            raise ValueError("v < r2 required")
        roi.InferredContinuousROM.__init__(self, "cAHGB",)
        self.r1, self.r2 = r1, r2
        self.v = v

    def _assemble_data_matrices(self, X_, U):
        """Construct the Operator Inference data matrices D1, D2
        from projected data:

        D1 = [1 | X_^T | (⊗2 X_)^T | U^T],
        D2 = [1 | X_^T | (⊗2 X_)^T | (⊗3 X21_)^T | U^T].

        Returns
        -------
        D1 : (k, 2 + r + r(r+1)/2) ndarray
            Operator Inference data matrix (non-temperature).

        D1 : (k, d + v(v+1)(v+2)/6) ndarray
            Operator Inference data matrix (temperature only).
        """
        k = X_.shape[1]
        d = 2 + self.r + self.r*(self.r + 1)//2

        # First data matrix: quadratic + input
        D = [np.ones((X_.shape[1],1)), X_.T, kron2c(X_).T, U.T]
        D1 = np.hstack(D)
        assert D1.shape == (k, d)

        # Second data matrix: cubic + input
        D.insert(-1, kron3c(X_[self.r1:(self.r1+self.v)]).T)
        D2 = np.hstack(D)
        assert D2.shape == (k, d + self.v*(self.v + 1)*(self.v + 2)//6)

        return D1, D2

    def _extract_operators(self, Os):
        """Extract and save the inferred operators from the block-matrix
        solution to the least-squarse problem.
        """
        # Unpack operators and check shapes.
        O1, O2, O3 = Os
        d1 = 2 + self.r + self.r*(self.r + 1)//2
        d2 = d1 + self.v*(self.v + 1)*(self.v + 2)//6
        assert O1.shape == (self.r1, d1)
        assert O2.shape == (self.v, d2)
        assert O3.shape == (self.r2 - self.v, d1)
        i = 0
        
        # Constant terms
        self.c_ = np.concatenate([O1[:,0], O2[:,0], O3[:,0]])
        assert self.c_.shape == (self.r,)
        i += 1
        
        # Linear terms
        self.A_ = np.row_stack([O1[:,i:i+self.r],
                                O2[:,i:i+self.r],
                                O3[:,i:i+self.r]])
        assert self.A_.shape == (self.r, self.r)
        i += self.r

        # (compact) Quadratic terms
        _r2 = self.r * (self.r + 1) // 2
        self.Hc_ = np.row_stack([O1[:,i:i+_r2],
                                 O2[:,i:i+_r2],
                                 O3[:,i:i+_r2]])
        assert self.Hc_.shape == (self.r, _r2)
        i += _r2

        # (compact) Cubic terms
        _r3 = self.v * (self.v + 1) * (self.v + 2) // 6
        self.Gc_ = O2[:,i:i+_r3]
        assert self.Gc_.shape == (self.v, _r3)
        
        # Linear input "matrix".
        self.B_ = np.concatenate([O1[:,-1],
                                  O2[:,-1],
                                  O3[:,-1]]).reshape((-1,1))
        assert self.B_.shape == (self.r, self.m)

        return

    def construct_solver(self, X_, rhs_, U):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        X_ : (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            projected to reduced order.

        rhs_ : (r,k) ndarray
            Time derivative training data. Each column is a snapshot,
            projected to reduced order.

        U : (1,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots.

        Returns
        -------
        self
        """
        X_, rhs_, U = self._process_fit_arguments(None, X_, rhs_, U)
        if self.r != self.r1 + self.r2:
            raise ValueError("reduced dimensions not aligned (r != r1 + r2)")
        D1, D2 = self._assemble_data_matrices(X_, U)
        Rs = np.split(rhs_.T, [self.r1, self.r1+self.v], axis=1)
        return MultiLstsqSolver().fit([D1, D2, D1], Rs,
                                      roi.lstsq.SolverTikhonov)

    def evaluate_solver(self, solver, λ1, λ2, λ3):
        """Evaluate the least-squares solver at the given regularization
        values, and save the resulting operators.
        """
        # Get block sizes for quadratic / cubic operators.
        r2 = self.r*(self.r + 1)//2
        r3 = self.v*(self.v + 1)*(self.v + 2)//6

        # Construct the regularization matrices.
        λa = np.zeros(2 + self.r + r2)
        λa[:(1 + self.r)] = λ1                          # Constant and linear.
        λa[(1 + self.r):(1 + self.r + r2)] = λ2         # Quadratic.
        λa[-1] = λ1                                     # Input.
        λb = np.concatenate([λa[:-1], np.full(r3, λ3), [λ1]])

        Otrps = solver.predict([λa, λb, λa])
        self._extract_operators([O.T for O in Otrps])

    def f_(self, t, x_, u):
        s = slice(self.r1, self.r1+self.v)
        x = self.c_ + self.A_ @ x_ + self.Hc_ @ kron2c(x_) + self.B_ @ u(t)
        x[s] += self.Gc_ @ kron3c(x_[s])
        return x

    def _construct_f_(self, *args, **kwargs):
        pass
