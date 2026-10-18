"""Exact sum-flux-driven ATS model for the existing Floquet infrastructure.

The Floquet solver, branch tracking, and displaced-state analysis remain those in
``FloquetAnalysis``.  This module only supplies the periodically driven ATS
Hamiltonian in the undriven ATS eigenbasis.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from .model import Model


class ATSModel(Model):
    r"""ATS driven through sum flux with differential flux held fixed.

    Energies are supplied as ordinary frequencies in GHz and converted once to
    angular frequency (rad/ns). ``drive_amplitudes`` is the standard normalized
    symmetric-flux coordinate ``Phi_Sigma/Phi0``:

        phi_sum(t) = phi_sum_dc + amplitude * cos(omega_d * t).

    The nonlinear flux dependence is retained exactly.  The Hamiltonian matches
    ``jaxquantum.ATS.get_H_nonlinear_static`` and is projected from ``n_pre_diag``
    oscillator states to the lowest ``n_ats`` undriven ATS eigenstates.
    """

    def __init__(
        self,
        ec_ghz: float,
        el_ghz: float,
        ej_ghz: float,
        dej_ghz: float,
        ej2_ghz: float,
        phi_sum_dc: float,
        phi_delta_dc: float,
        omega_d_values: np.ndarray,
        drive_amplitudes: np.ndarray,
        n_ats: int = 16,
        n_pre_diag: int = 80,
    ):
        from jaxquantum.core.settings import SETTINGS
        import jaxquantum as jqt
        from jaxquantum.devices import ATS

        SETTINGS["auto_tidyup_atol"] = 0.0

        self.ec_ghz = float(ec_ghz)
        self.el_ghz = float(el_ghz)
        self.ej_ghz = float(ej_ghz)
        self.dej_ghz = float(dej_ghz)
        self.ej2_ghz = float(ej2_ghz)
        self.phi_sum_dc = float(phi_sum_dc)
        self.phi_delta_dc = float(phi_delta_dc)
        self.n_ats = int(n_ats)
        self.n_pre_diag = int(n_pre_diag)

        if self.n_pre_diag < self.n_ats:
            raise ValueError("n_pre_diag must be greater than or equal to n_ats")

        twopi = 2.0 * np.pi
        ec = twopi * self.ec_ghz
        el = twopi * self.el_ghz
        ej = twopi * self.ej_ghz
        dej = twopi * self.dej_ghz
        ej2 = twopi * self.ej2_ghz

        ats = ATS.create(
            self.n_ats,
            {
                "Ec": ec,
                "El": el,
                "Ej": ej,
                "dEj": dej,
                "Ej2": ej2,
                "phi_sum_ext": self.phi_sum_dc,
                "phi_delta_ext": self.phi_delta_dc,
            },
            N_pre_diag=self.n_pre_diag,
            use_linear=False,
        )

        evals, evecs = ats.eig_systems["vals"], ats.eig_systems["vecs"]
        evals = np.asarray(evals[: self.n_ats])
        evecs = np.asarray(evecs[:, : self.n_ats])

        def project(op) -> qt.Qobj:
            matrix = evecs.conj().T @ np.asarray(op.data) @ evecs
            return qt.Qobj(matrix)

        phi = ats.linear_ops["phi"]
        cos_phi = jqt.cosm(phi)
        sin_phi = jqt.sinm(phi)
        cos_2phi = cos_phi @ cos_phi - sin_phi @ sin_phi
        sin_2phi = 2.0 * cos_phi @ sin_phi

        delta = twopi * self.phi_delta_dc
        delta2 = 2.0 * delta
        shifted_cos_phi = np.cos(delta) * project(cos_phi) - np.sin(delta) * project(sin_phi)
        shifted_sin_phi = np.cos(delta) * project(sin_phi) + np.sin(delta) * project(cos_phi)
        shifted_cos_2phi = np.cos(delta2) * project(cos_2phi) - np.sin(delta2) * project(sin_2phi)

        self._op_ej = -2.0 * ej * shifted_cos_phi
        self._op_dej = 2.0 * dej * shifted_sin_phi
        self._op_ej2 = 2.0 * ej2 * shifted_cos_2phi

        h0 = qt.Qobj(np.diag(evals - evals[0]))
        super().__init__(
            H0=h0,
            H1=qt.Qobj(np.zeros_like(h0.full())),
            omega_d_values=omega_d_values,
            drive_amplitudes=drive_amplitudes,
        )

    def hamiltonian(self, omega_d_amp: tuple[float, float]) -> qt.QobjEvo:
        """Return the exact periodic Hamiltonian for one frequency/amplitude pair."""
        omega_d, amplitude = (float(value) for value in omega_d_amp)
        phi_sum_dc = self.phi_sum_dc

        def phase(t: float) -> float:
            return 2.0 * np.pi * (
                phi_sum_dc + amplitude * np.cos(omega_d * t)
            )

        cos_dc = np.cos(2.0 * np.pi * phi_sum_dc)
        sin_dc = np.sin(2.0 * np.pi * phi_sum_dc)
        cos2_dc = np.cos(4.0 * np.pi * phi_sum_dc)

        terms: list = [self.H0]
        if self.ej_ghz != 0.0:
            terms.append([self._op_ej, lambda t: np.cos(phase(t)) - cos_dc])
        if self.dej_ghz != 0.0:
            terms.append([self._op_dej, lambda t: np.sin(phase(t)) - sin_dc])
        if self.ej2_ghz != 0.0:
            terms.append([self._op_ej2, lambda t: np.cos(2.0 * phase(t)) - cos2_dc])
        return qt.QobjEvo(terms)


class ExactSumFluxModel(Model):
    """Generic exact sum-flux model in a user-supplied static eigenbasis.

    This is intended for a hierarchically diagonalized composite system.  It does
    not diagonalize, track, fit, or interpret Floquet modes; those operations remain
    in :class:`floquet.FloquetAnalysis`. As in :class:`ATSModel`,
    ``drive_amplitudes`` is normalized ``Phi_Sigma/Phi0``.
    """

    def __init__(
        self,
        H0: qt.Qobj | np.ndarray,
        cos_phi: qt.Qobj | np.ndarray,
        sin_phi: qt.Qobj | np.ndarray,
        cos_2phi: qt.Qobj | np.ndarray,
        sin_2phi: qt.Qobj | np.ndarray,
        ej_rad_per_ns: float,
        dej_rad_per_ns: float,
        ej2_rad_per_ns: float,
        phi_sum_dc: float,
        phi_delta_dc: float,
        omega_d_values: np.ndarray,
        drive_amplitudes: np.ndarray,
    ):
        def qobj(value) -> qt.Qobj:
            return value if isinstance(value, qt.Qobj) else qt.Qobj(value)

        self.cos_phi = qobj(cos_phi)
        self.sin_phi = qobj(sin_phi)
        self.cos_2phi = qobj(cos_2phi)
        self.sin_2phi = qobj(sin_2phi)
        self.ej_rad_per_ns = float(ej_rad_per_ns)
        self.dej_rad_per_ns = float(dej_rad_per_ns)
        self.ej2_rad_per_ns = float(ej2_rad_per_ns)
        self.phi_sum_dc = float(phi_sum_dc)
        self.phi_delta_dc = float(phi_delta_dc)

        delta = 2.0 * np.pi * self.phi_delta_dc
        shifted_cos_phi = np.cos(delta) * self.cos_phi - np.sin(delta) * self.sin_phi
        shifted_sin_phi = np.cos(delta) * self.sin_phi + np.sin(delta) * self.cos_phi
        shifted_cos_2phi = np.cos(2.0 * delta) * self.cos_2phi - np.sin(2.0 * delta) * self.sin_2phi
        self._op_ej = -2.0 * self.ej_rad_per_ns * shifted_cos_phi
        self._op_dej = 2.0 * self.dej_rad_per_ns * shifted_sin_phi
        self._op_ej2 = 2.0 * self.ej2_rad_per_ns * shifted_cos_2phi

        H0 = qobj(H0)
        super().__init__(
            H0=H0,
            H1=qt.Qobj(np.zeros_like(H0.full())),
            omega_d_values=omega_d_values,
            drive_amplitudes=drive_amplitudes,
        )

    def hamiltonian(self, omega_d_amp: tuple[float, float]) -> qt.QobjEvo:
        omega_d, amplitude = (float(value) for value in omega_d_amp)
        phi_sum_dc = self.phi_sum_dc

        def phase(t: float) -> float:
            return 2.0 * np.pi * (phi_sum_dc + amplitude * np.cos(omega_d * t))

        cos_dc = np.cos(2.0 * np.pi * phi_sum_dc)
        sin_dc = np.sin(2.0 * np.pi * phi_sum_dc)
        cos2_dc = np.cos(4.0 * np.pi * phi_sum_dc)
        terms: list = [self.H0]
        if self.ej_rad_per_ns != 0.0:
            terms.append([self._op_ej, lambda t: np.cos(phase(t)) - cos_dc])
        if self.dej_rad_per_ns != 0.0:
            terms.append([self._op_dej, lambda t: np.sin(phase(t)) - sin_dc])
        if self.ej2_rad_per_ns != 0.0:
            terms.append([self._op_ej2, lambda t: np.cos(2.0 * phase(t)) - cos2_dc])
        return qt.QobjEvo(terms)

