"""Local full-configuration-space stability tests for a single rigid origami vertex.

The core kinematics follow the product-of-rotations convention used in
Eqs. (2.1)--(2.3) of the manuscript.  For a compatible state phi_star,
the columns of the deformed crease-axis matrix C are the Jacobian of the
local SO(3) loop-closure residual.  Thus ker(C) is the tangent space of
infinitesimally compatible fold-angle perturbations at a regular state.

A tangent vector alone is not a finite compatible perturbation.  This
module therefore retracts tangent perturbations back to the nonlinear
loop-closure manifold and computes the gradient/Hessian of the resulting
reduced energy numerically.  This captures the curvature of the
compatibility manifold, which is essential for detecting saddles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import null_space
from scipy.optimize import root
from scipy.spatial.transform import Rotation

FloatArray = NDArray[np.float64]


def _as_float_vector(x: ArrayLike, *, length: int | None = None, name: str = "x") -> FloatArray:
    out = np.asarray(x, dtype=float).reshape(-1)
    if length is not None and out.size != length:
        raise ValueError(f"{name} must have length {length}; got {out.size}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite entries.")
    return out


def _validate_crease_matrix(B: ArrayLike) -> FloatArray:
    out = np.asarray(B, dtype=float)
    if out.ndim != 2 or out.shape[0] != 3:
        raise ValueError(f"B must have shape (3, N); got {out.shape}.")
    if out.shape[1] < 4:
        raise ValueError("A vertex must have at least four creases.")
    norms = np.linalg.norm(out, axis=0)
    if np.any(norms <= 0.0):
        raise ValueError("Every crease vector must be nonzero.")
    out = out / norms
    if not np.all(np.isfinite(out)):
        raise ValueError("B contains non-finite entries.")
    return out


def crease_vectors_degree6(alpha1: float) -> FloatArray:
    """Return the 3 x 6 flat-state crease-axis matrix for Sec. 3(a).

    The coordinate convention is b1 = e1 and the creases are numbered
    counterclockwise in the e1-e2 plane.  The sector-angle sequence is

        [alpha1, pi - 2 alpha1, alpha1,
         alpha1, pi - 2 alpha1, alpha1].
    """
    a = float(alpha1)
    if not (0.0 < a < 0.5 * np.pi):
        raise ValueError("degree-6 geometry requires 0 < alpha1 < pi/2.")
    theta = np.array([0.0, a, np.pi - a, np.pi, np.pi + a, 2.0 * np.pi - a])
    return np.vstack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))


def crease_vectors_degree8(alpha1: float) -> FloatArray:
    """Return the 3 x 8 flat-state crease-axis matrix for Sec. 3(b).

    This construction uses the vector relations stated in the manuscript:
    b1 = -b5 = e1, b3 = -b7 = e2, b6 = sigma_e2 b4, and
    b8 = sigma_e2 b2.  With counterclockwise numbering, the polar angles are

        [0, alpha1, pi/2, pi-alpha1,
         pi, pi+alpha1, 3pi/2, 2pi-alpha1].

    Consequently the sector-angle sequence is

        [alpha1, pi/2-alpha1, pi/2-alpha1, alpha1,
         alpha1, pi/2-alpha1, pi/2-alpha1, alpha1].
    """
    a = float(alpha1)
    if not (0.0 < a < 0.5 * np.pi):
        raise ValueError("degree-8 geometry requires 0 < alpha1 < pi/2.")
    theta = np.array(
        [
            0.0,
            a,
            0.5 * np.pi,
            np.pi - a,
            np.pi,
            np.pi + a,
            1.5 * np.pi,
            2.0 * np.pi - a,
        ]
    )
    return np.vstack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))



def assemble_symmetric_folds_degree6(
    phi1: float, phi2: float, phi3: float, phi4: float
) -> FloatArray:
    """Assemble the D1 fold pattern [phi1, phi2, phi3, phi4, phi3, phi2]."""
    return np.array([phi1, phi2, phi3, phi4, phi3, phi2], dtype=float)


def assemble_symmetric_folds_degree8(
    phi1: float, phi2: float, phi3: float
) -> FloatArray:
    """Assemble the D2 fold pattern [phi1, phi2, phi3, phi2, phi1, phi2, phi3, phi2]."""
    return np.array([phi1, phi2, phi3, phi2, phi1, phi2, phi3, phi2], dtype=float)

def rotation_matrix(axis: ArrayLike, angle: float) -> FloatArray:
    """Rodrigues rotation matrix for a right-handed rotation about axis."""
    u = _as_float_vector(axis, length=3, name="axis")
    norm = np.linalg.norm(u)
    if norm <= 0.0:
        raise ValueError("axis must be nonzero.")
    u = u / norm
    ux, uy, uz = u
    cross = np.array(
        [[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]], dtype=float
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


@dataclass(frozen=True)
class ForwardKinematics:
    """Result of walking once around a vertex."""

    closure_rotation: FloatArray  # shape (3, 3); identity when compatible
    deformed_creases: FloatArray  # shape (3, N), columns c_i = F_{i-1} b_i
    prefix_rotations: FloatArray  # shape (N+1, 3, 3), F_0 through F_N


def forward_kinematics(B: ArrayLike, phi: ArrayLike) -> ForwardKinematics:
    """Compute deformed crease axes and the loop-closure rotation.

    Matrices act on column vectors.  The update is

        F_i = F_{i-1} Q_{b_i}(phi_i),
        c_i = F_{i-1} b_i.

    Thus F_N = I is the nonlinear compatibility condition.
    """
    Bm = _validate_crease_matrix(B)
    angles = _as_float_vector(phi, length=Bm.shape[1], name="phi")

    n = Bm.shape[1]
    prefixes = np.empty((n + 1, 3, 3), dtype=float)
    C = np.empty_like(Bm)
    F = np.eye(3)
    prefixes[0] = F

    for i in range(n):
        C[:, i] = F @ Bm[:, i]
        F = F @ rotation_matrix(Bm[:, i], angles[i])
        prefixes[i + 1] = F

    return ForwardKinematics(F, C, prefixes)


def closure_residual(B: ArrayLike, phi: ArrayLike) -> FloatArray:
    """Return the local SO(3) loop-closure residual as a rotation vector.

    The residual is smooth in a neighborhood of a compatible state because
    the closure rotation is then near the identity.  Its Jacobian at exact
    closure is the deformed crease-axis matrix returned by
    :func:`forward_kinematics`.
    """
    F = forward_kinematics(B, phi).closure_rotation
    return Rotation.from_matrix(F).as_rotvec()


@dataclass(frozen=True)
class TangentSpaces:
    """SVD-based tangent/normal decomposition at a compatible state."""

    jacobian: FloatArray  # 3 x N; columns are deformed crease axes
    tangent_basis: FloatArray  # N x (N-rank), orthonormal basis Z for ker(J)
    normal_basis: FloatArray  # N x rank, orthonormal row-space basis Y
    singular_values: FloatArray
    rank: int


def tangent_spaces(
    B: ArrayLike,
    phi: ArrayLike,
    *,
    relative_tolerance: float = 1.0e-10,
    closure_tolerance: float = 1.0e-8,
) -> TangentSpaces:
    """Compute infinitesimally compatible and normal fold-angle spaces.

    At a regular compatible vertex, rank(J)=3 and dim ker(J)=N-3.  At a
    singular configuration (notably the flat state), rank can drop; the
    first-order nullspace then need not integrate to finite folding paths.
    """
    Bm = _validate_crease_matrix(B)
    angles = _as_float_vector(phi, length=Bm.shape[1], name="phi")
    kin = forward_kinematics(Bm, angles)
    residual_norm = np.linalg.norm(Rotation.from_matrix(kin.closure_rotation).as_rotvec())
    if residual_norm > closure_tolerance:
        raise ValueError(
            f"State is not compatible: closure residual norm = {residual_norm:.3e}."
        )

    _, singular_values, Vt = np.linalg.svd(kin.deformed_creases, full_matrices=True)
    threshold = relative_tolerance * max(1.0, singular_values[0])
    rank = int(np.count_nonzero(singular_values > threshold))
    V = Vt.T
    Y = V[:, :rank]
    Z = V[:, rank:]
    return TangentSpaces(kin.deformed_creases, Z, Y, singular_values, rank)


@dataclass(frozen=True)
class RetractionResult:
    phi: FloatArray
    normal_coordinates: FloatArray
    closure_norm: float
    success: bool
    message: str


def retract_to_compatibility(
    B: ArrayLike,
    phi_star: ArrayLike,
    tangent_basis: ArrayLike,
    normal_basis: ArrayLike,
    q: ArrayLike,
    *,
    eta_initial: ArrayLike | None = None,
    closure_tolerance: float = 1.0e-10,
    max_function_evaluations: int = 200,
) -> RetractionResult:
    """Retract a tangent perturbation onto the exact closure manifold.

    The local coordinates are

        phi(q, eta) = phi_star + Z q + Y eta,

    where Z spans ker(J) and Y spans row(J).  For a regular state, the
    implicit-function theorem guarantees a local eta(q) satisfying closure.
    """
    Bm = _validate_crease_matrix(B)
    phi0 = _as_float_vector(phi_star, length=Bm.shape[1], name="phi_star")
    Z = np.asarray(tangent_basis, dtype=float)
    Y = np.asarray(normal_basis, dtype=float)
    if Z.ndim != 2 or Z.shape[0] != Bm.shape[1]:
        raise ValueError("tangent_basis must have shape (N, d).")
    if Y.ndim != 2 or Y.shape[0] != Bm.shape[1]:
        raise ValueError("normal_basis must have shape (N, r).")
    qv = _as_float_vector(q, length=Z.shape[1], name="q")
    if eta_initial is None:
        eta0 = np.zeros(Y.shape[1], dtype=float)
    else:
        eta0 = _as_float_vector(eta_initial, length=Y.shape[1], name="eta_initial")

    def residual_eta(eta: FloatArray) -> FloatArray:
        trial = phi0 + Z @ qv + Y @ eta
        return closure_residual(Bm, trial)

    solution = root(
        residual_eta,
        eta0,
        method="hybr",
        options={"maxfev": int(max_function_evaluations), "xtol": closure_tolerance},
    )
    eta = np.asarray(solution.x, dtype=float)
    phi = phi0 + Z @ qv + Y @ eta
    norm = float(np.linalg.norm(closure_residual(Bm, phi)))
    # Some MINPACK runs report "no progress" after already reaching machine-
    # precision closure.  The residual norm, not the status flag, is the
    # decisive criterion for this local projection.
    success = bool(norm <= 10.0 * closure_tolerance)
    return RetractionResult(phi, eta, norm, success, str(solution.message))


def torsional_energy(phi: ArrayLike, stiffness: ArrayLike, rest_angles: ArrayLike) -> float:
    """Linear torsional crease energy, Eq. (3.1)."""
    angles = _as_float_vector(phi, name="phi")
    k = _as_float_vector(stiffness, length=angles.size, name="stiffness")
    phi0 = _as_float_vector(rest_angles, length=angles.size, name="rest_angles")
    if np.any(k <= 0.0):
        raise ValueError("All crease stiffnesses must be positive.")
    delta = angles - phi0
    return float(0.5 * np.dot(k * delta, delta))


class RetractionFailure(RuntimeError):
    pass


def make_reduced_energy(
    B: ArrayLike,
    phi_star: ArrayLike,
    stiffness: ArrayLike,
    rest_angles: ArrayLike,
    tangent_basis: ArrayLike,
    normal_basis: ArrayLike,
    *,
    closure_tolerance: float = 1.0e-10,
) -> Callable[[FloatArray], float]:
    """Create E(q)=U(phi(q)) on the exact local compatibility manifold."""
    Bm = _validate_crease_matrix(B)
    phi0 = _as_float_vector(phi_star, length=Bm.shape[1], name="phi_star")
    k = _as_float_vector(stiffness, length=Bm.shape[1], name="stiffness")
    rest = _as_float_vector(rest_angles, length=Bm.shape[1], name="rest_angles")
    Z = np.asarray(tangent_basis, dtype=float)
    Y = np.asarray(normal_basis, dtype=float)

    def reduced_energy(q: FloatArray) -> float:
        result = retract_to_compatibility(
            Bm,
            phi0,
            Z,
            Y,
            q,
            closure_tolerance=closure_tolerance,
        )
        if not result.success:
            raise RetractionFailure(
                f"Compatibility retraction failed (norm={result.closure_norm:.3e}): "
                f"{result.message}"
            )
        return torsional_energy(result.phi, k, rest)

    return reduced_energy


@dataclass(frozen=True)
class NumericalDerivatives:
    value: float
    gradient: FloatArray
    hessian: FloatArray


def central_gradient_hessian(
    function: Callable[[FloatArray], float],
    dimension: int,
    *,
    step: float = 1.0e-3,
) -> NumericalDerivatives:
    """Central finite-difference gradient and Hessian at q=0."""
    if dimension < 1:
        raise ValueError("dimension must be at least one.")
    if step <= 0.0:
        raise ValueError("step must be positive.")

    zero = np.zeros(dimension, dtype=float)
    f0 = float(function(zero))
    gradient = np.empty(dimension, dtype=float)
    hessian = np.empty((dimension, dimension), dtype=float)
    eye = np.eye(dimension)

    plus: list[float] = []
    minus: list[float] = []
    for i in range(dimension):
        fp = float(function(step * eye[i]))
        fm = float(function(-step * eye[i]))
        plus.append(fp)
        minus.append(fm)
        gradient[i] = (fp - fm) / (2.0 * step)
        hessian[i, i] = (fp - 2.0 * f0 + fm) / (step * step)

    for i in range(dimension):
        for j in range(i + 1, dimension):
            fpp = float(function(step * (eye[i] + eye[j])))
            fpm = float(function(step * (eye[i] - eye[j])))
            fmp = float(function(step * (-eye[i] + eye[j])))
            fmm = float(function(-step * (eye[i] + eye[j])))
            hij = (fpp - fpm - fmp + fmm) / (4.0 * step * step)
            hessian[i, j] = hij
            hessian[j, i] = hij

    hessian = 0.5 * (hessian + hessian.T)
    return NumericalDerivatives(f0, gradient, hessian)


def symmetry_equalities_degree6() -> FloatArray:
    """A phi = 0 encodes D1-preserving infinitesimal fold-angle changes."""
    A = np.zeros((2, 6), dtype=float)
    A[0, 4] = 1.0  # delta phi5 - delta phi3 = 0
    A[0, 2] = -1.0
    A[1, 5] = 1.0  # delta phi6 - delta phi2 = 0
    A[1, 1] = -1.0
    return A


def symmetry_equalities_degree8() -> FloatArray:
    """A phi = 0 encodes the D2 fold-angle pattern [a,b,c,b,a,b,c,b]."""
    A = np.zeros((5, 8), dtype=float)
    pairs = [(3, 1), (4, 0), (5, 1), (6, 2), (7, 1)]
    for row, (i, j) in enumerate(pairs):
        A[row, i] = 1.0
        A[row, j] = -1.0
    return A


@dataclass(frozen=True)
class SymmetryDecomposition:
    symmetric_q_basis: FloatArray
    breaking_q_basis: FloatArray


def decompose_tangent_by_symmetry(
    tangent_basis: ArrayLike,
    symmetry_equalities: ArrayLike,
    *,
    tolerance: float = 1.0e-10,
) -> SymmetryDecomposition:
    """Split tangent coordinates into symmetry-preserving and orthogonal parts."""
    Z = np.asarray(tangent_basis, dtype=float)
    A = np.asarray(symmetry_equalities, dtype=float)
    if Z.ndim != 2 or A.ndim != 2 or A.shape[1] != Z.shape[0]:
        raise ValueError("Incompatible tangent_basis and symmetry_equalities shapes.")

    Qsym = null_space(A @ Z, rcond=tolerance)
    if Qsym.shape[1] == 0:
        Qbreak = np.eye(Z.shape[1])
    elif Qsym.shape[1] == Z.shape[1]:
        Qbreak = np.zeros((Z.shape[1], 0), dtype=float)
    else:
        Qbreak = null_space(Qsym.T, rcond=tolerance)
    return SymmetryDecomposition(Qsym, Qbreak)


Status = Literal[
    "strict_local_minimum",
    "saddle",
    "not_stationary",
    "marginal_or_inconclusive",
    "singular_configuration",
]


@dataclass(frozen=True)
class StabilityResult:
    status: Status
    closure_norm: float
    constraint_rank: int
    tangent_dimension: int
    energy: float
    reduced_gradient: FloatArray
    reduced_hessian: FloatArray
    eigenvalues: FloatArray
    eigenvectors_q: FloatArray
    most_unstable_fold_mode: FloatArray | None
    finite_difference_step: float
    tangent_basis: FloatArray


def classify_relaxed_stability(
    B: ArrayLike,
    phi_star: ArrayLike,
    stiffness: ArrayLike,
    rest_angles: ArrayLike,
    *,
    finite_difference_step: float = 1.0e-3,
    gradient_tolerance: float = 1.0e-6,
    eigenvalue_tolerance: float = 1.0e-6,
    closure_tolerance: float = 1.0e-10,
) -> StabilityResult:
    """Classify local stability while *ignoring all self-contact constraints*.

    A strict minimum in this enlarged compatible space is a certified lower-
    bound survivor: adding contact constraints cannot destabilize it.  A
    negative result does not prove physical instability because the descent
    path may require facet penetration.
    """
    Bm = _validate_crease_matrix(B)
    phi0 = _as_float_vector(phi_star, length=Bm.shape[1], name="phi_star")
    closure_norm = float(np.linalg.norm(closure_residual(Bm, phi0)))
    spaces = tangent_spaces(
        Bm,
        phi0,
        closure_tolerance=max(closure_tolerance, 10.0 * closure_norm + 1.0e-12),
    )
    d = spaces.tangent_basis.shape[1]
    energy0 = torsional_energy(phi0, stiffness, rest_angles)

    if spaces.rank != 3:
        return StabilityResult(
            "singular_configuration",
            closure_norm,
            spaces.rank,
            d,
            energy0,
            np.full(d, np.nan),
            np.full((d, d), np.nan),
            np.full(d, np.nan),
            np.full((d, d), np.nan),
            None,
            finite_difference_step,
            spaces.tangent_basis,
        )

    reduced_energy = make_reduced_energy(
        Bm,
        phi0,
        stiffness,
        rest_angles,
        spaces.tangent_basis,
        spaces.normal_basis,
        closure_tolerance=closure_tolerance,
    )
    derivatives = central_gradient_hessian(
        reduced_energy, d, step=finite_difference_step
    )
    eigenvalues, eigenvectors = np.linalg.eigh(derivatives.hessian)

    gradient_norm = float(np.linalg.norm(derivatives.gradient))
    if gradient_norm > gradient_tolerance:
        status: Status = "not_stationary"
        direction_q = -derivatives.gradient / gradient_norm
        mode = spaces.tangent_basis @ direction_q
    elif eigenvalues[0] < -eigenvalue_tolerance:
        status = "saddle"
        mode = spaces.tangent_basis @ eigenvectors[:, 0]
    elif eigenvalues[0] > eigenvalue_tolerance:
        status = "strict_local_minimum"
        mode = None
    else:
        status = "marginal_or_inconclusive"
        mode = spaces.tangent_basis @ eigenvectors[:, 0]

    return StabilityResult(
        status,
        closure_norm,
        spaces.rank,
        d,
        derivatives.value,
        derivatives.gradient,
        derivatives.hessian,
        eigenvalues,
        eigenvectors,
        mode,
        finite_difference_step,
        spaces.tangent_basis,
    )


@dataclass(frozen=True)
class SymmetryBreakingResult:
    full_result: StabilityResult
    symmetric_dimension: int
    breaking_dimension: int
    symmetric_gradient: FloatArray
    breaking_gradient: FloatArray
    symmetric_hessian: FloatArray
    breaking_hessian: FloatArray
    coupling_hessian: FloatArray
    breaking_eigenvalues: FloatArray
    most_unstable_breaking_fold_mode: FloatArray | None


def analyze_symmetry_breaking(
    B: ArrayLike,
    phi_star: ArrayLike,
    stiffness: ArrayLike,
    rest_angles: ArrayLike,
    symmetry_equalities: ArrayLike,
    *,
    finite_difference_step: float = 1.0e-3,
    gradient_tolerance: float = 1.0e-6,
    eigenvalue_tolerance: float = 1.0e-6,
    closure_tolerance: float = 1.0e-10,
) -> SymmetryBreakingResult:
    """Resolve the reduced derivatives into symmetry-preserving/breaking blocks."""
    Bm = _validate_crease_matrix(B)
    phi0 = _as_float_vector(phi_star, length=Bm.shape[1], name="phi_star")
    full = classify_relaxed_stability(
        Bm,
        phi0,
        stiffness,
        rest_angles,
        finite_difference_step=finite_difference_step,
        gradient_tolerance=gradient_tolerance,
        eigenvalue_tolerance=eigenvalue_tolerance,
        closure_tolerance=closure_tolerance,
    )

    if full.constraint_rank != 3:
        empty = np.empty((0,), dtype=float)
        empty2 = np.empty((0, 0), dtype=float)
        return SymmetryBreakingResult(
            full, 0, 0, empty, empty, empty2, empty2, empty2, empty, None
        )

    Z = full.tangent_basis
    split = decompose_tangent_by_symmetry(Z, symmetry_equalities)
    Qs = split.symmetric_q_basis
    Qb = split.breaking_q_basis
    g = full.reduced_gradient
    H = full.reduced_hessian

    gs = Qs.T @ g
    gb = Qb.T @ g
    Hss = Qs.T @ H @ Qs
    Hbb = Qb.T @ H @ Qb
    Hsb = Qs.T @ H @ Qb

    if Qb.shape[1] == 0:
        evals = np.empty((0,), dtype=float)
        mode = None
    else:
        evals, evecs = np.linalg.eigh(Hbb)
        if np.linalg.norm(gb) > gradient_tolerance:
            qb = Qb @ (-gb / np.linalg.norm(gb))
            mode = Z @ qb
        elif evals[0] <= eigenvalue_tolerance:
            qb = Qb @ evecs[:, 0]
            mode = Z @ qb
        else:
            mode = None

    return SymmetryBreakingResult(
        full,
        Qs.shape[1],
        Qb.shape[1],
        gs,
        gb,
        Hss,
        Hbb,
        Hsb,
        evals,
        mode,
    )


def compatible_mode_scan(
    B: ArrayLike,
    phi_star: ArrayLike,
    stiffness: ArrayLike,
    rest_angles: ArrayLike,
    fold_mode: ArrayLike,
    amplitudes: ArrayLike,
    *,
    closure_tolerance: float = 1.0e-10,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Trace a selected tangent mode with exact closure retraction.

    Returns (amplitudes, energies, fold_angles), where fold_angles has shape
    (len(amplitudes), N).  The supplied fold_mode is projected into the
    tangent space and normalized before continuation.
    """
    Bm = _validate_crease_matrix(B)
    phi0 = _as_float_vector(phi_star, length=Bm.shape[1], name="phi_star")
    mode = _as_float_vector(fold_mode, length=Bm.shape[1], name="fold_mode")
    amps = _as_float_vector(amplitudes, name="amplitudes")
    spaces = tangent_spaces(Bm, phi0)
    if spaces.rank != 3:
        raise ValueError("Mode scanning currently requires a regular rank-3 state.")

    q_direction = spaces.tangent_basis.T @ mode
    norm = np.linalg.norm(q_direction)
    if norm <= 1.0e-14:
        raise ValueError("fold_mode has no component in the compatible tangent space.")
    q_direction = q_direction / norm

    energies = np.empty(amps.size, dtype=float)
    states = np.empty((amps.size, Bm.shape[1]), dtype=float)
    for i, amplitude in enumerate(amps):
        result = retract_to_compatibility(
            Bm,
            phi0,
            spaces.tangent_basis,
            spaces.normal_basis,
            amplitude * q_direction,
            closure_tolerance=closure_tolerance,
        )
        if not result.success:
            raise RetractionFailure(
                f"Retraction failed at amplitude {amplitude:.6g}: {result.message}"
            )
        states[i] = result.phi
        energies[i] = torsional_energy(result.phi, stiffness, rest_angles)

    return amps, energies, states
