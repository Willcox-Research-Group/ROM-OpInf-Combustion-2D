# combustion_rom.py
"""Custom Operator Inference ROM class for the combustion problem,
with a separate basis for non-temperature variables and temperature,
with a quadratic model for the non-temperature equations and a
cubic model for the temperature equations.
"""
import numpy as np

import rom_operator_inference as roi


# Utilities ===================================================================

class MultiLstsqSolver(roi.lstsq._tikhonov._BaseLstsqSolver):
    """Encapsulate multiple least-squares solvers."""
    def fit(self, As, Bs, SolverClass=roi.lstsq.LstsqSolverL2):
        """Initialize and fit a solver for each pair of inputs."""
        # Check inputs.
        if len(As) != len(Bs):
            raise ValueError("inputs not aligned")
        if not issubclass(SolverClass, roi.lstsq._tikhonov._BaseLstsqSolver):
            raise TypeError("invalid SolverClass")

        # Fit each solver.
        self.solvers_ = [SolverClass(compute_extras=False).fit(A,B)
                         for A,B in zip(As, Bs)]
        return self

    def predict(self, P):
        """Predict with each solver."""
        return [solver.predict(P) for solver in self.solvers_]


class CombustionROM(roi.InferredContinuousROM):
    """Operator Inference ROM with a specific structure:
    * cAHB (quadratic with input) for the first r1 modes
    * cAHGB (cubic with input) for the last r2 modes
    """
    def __init__(self, r1, r2):
        roi.InferredContinuousROM.__init__(self, "cAHGB",)
        self.r1, self.r2 = r1, r2

    def _assemble_data_matrices(self, X_, U):
        """Construct the Operator Inference data matrices D1, D2
        from projected data:

        D1 = [1 | X_^T | (⊗2 X_)^T | U^T],
        D2 = [1 | X_^T | (⊗2 X_)^T | (⊗3 X_)^T | U^T].

        Returns
        -------
        D1 : (k, 2 + r + r(r+1)/2) ndarray
            Operator Inference data matrix (non-temperature).

        D1 : (k, d + r(r+1)(r+2)/6) ndarray
            Operator Inference data matrix (temperature only).
        """
        k = X_.shape[1]
        d = 2 + self.r + self.r*(self.r + 1)//2

        # First data matrix: quadratic + input
        D = [np.ones((X_.shape[1],1)), X_.T, roi.utils.kron2c(X_).T, U.T]
        D1 = np.hstack(D)
        assert D1.shape == (k, d)

        # Second data matrix: cubic + input
        D.insert(-1, roi.utils.kron3c(X_).T)
        D2 = np.hstack(D)
        assert D2.shape == (k, d + self.r*(self.r + 1)*(self.r + 2)//6)

        return D1, D2

    def _extract_operators(self, Os):
        """Extract and save the inferred operators from the block-matrix
        solution to the least-squarse problem.
        """
        # Unpack operators and check shapes.
        O1, O2 = Os
        d1 = 2 + self.r + self.r*(self.r + 1)//2
        d2 = d1 + self.r*(self.r + 1)*(self.r + 2)//6
        assert O1.shape == (self.r1, d1)
        assert O2.shape == (self.r2, d2)
        i = 0
        
        # Constant terms
        self.c_ = np.concatenate([O1[:,0], O2[:,0]])
        assert self.c_.shape == (self.r,)
        i += 1
        
        # Linear terms
        self.A_ = np.row_stack([O1[:,i:i+self.r], O2[:,i:i+self.r]])
        assert self.A_.shape == (self.r, self.r)
        i += self.r

        # (compact) Quadratic terms
        _r2 = self.r * (self.r + 1) // 2
        self.Hc_ = np.row_stack([O1[:,i:i+_r2], O2[:,i:i+_r2]])
        assert self.Hc_.shape == (self.r, _r2)
        i += _r2

        # (compact) Cubic terms
        _r3 = self.r * (self.r + 1) * (self.r + 2) // 6
        G2 = O2[:,i:i+_r3]
        self.Gc_ = np.row_stack([np.zeros((self.r1,_r3)), G2])
        assert self.Gc_.shape == (self.r, _r3)
        
        # Linear input "matrix".
        self.B_ = np.concatenate([O1[:,-1], O2[:,-1]]).reshape((-1,1))
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
        Ds = self._assemble_data_matrices(X_, U)
        Rs = np.split(rhs_.T, [self.r1], axis=1)
        self.solver_ = MultiLstsqSolver().fit(Ds, Rs, roi.lstsq.LstsqSolverL2)
        return self

    def evaluate_solver(self, λ):
        Otrps = self.solver_.predict(λ)
        self._extract_operators([O.T for O in Otrps])
