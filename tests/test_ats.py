import numpy as np
import qutip as qt

from floquet import ATSModel, FloquetAnalysis, Options


def make_model(n_ats=12, n_pre_diag=60):
    return ATSModel(
        ec_ghz=0.1742,
        el_ghz=21.07,
        ej_ghz=9.0,
        dej_ghz=0.16,
        ej2_ghz=0.0,
        phi_sum_dc=0.25,
        phi_delta_dc=0.0,
        omega_d_values=2 * np.pi * np.array([0.4, 6.4]),
        drive_amplitudes=np.array([0.0, 0.02125]),
        n_ats=n_ats,
        n_pre_diag=n_pre_diag,
    )


def test_zero_drive_is_h0():
    model = make_model()
    for frequency in model.omega_d_values:
        for fraction in (0.0, 0.17, 0.83):
            period = 2 * np.pi / frequency
            assert np.allclose(
                model.hamiltonian((frequency, 0.0))(fraction * period).full(),
                model.H0.full(),
                atol=1e-12,
            )


def test_exact_matrix_matches_direct_jaxquantum_projection():
    from jaxquantum.core.settings import SETTINGS
    import jaxquantum as jqt
    from jaxquantum.devices import ATS

    SETTINGS["auto_tidyup_atol"] = 0.0
    model = make_model()
    twopi = 2 * np.pi
    params = {
        "Ec": twopi * model.ec_ghz,
        "El": twopi * model.el_ghz,
        "Ej": twopi * model.ej_ghz,
        "dEj": twopi * model.dej_ghz,
        "Ej2": twopi * model.ej2_ghz,
        "phi_sum_ext": model.phi_sum_dc,
        "phi_delta_ext": model.phi_delta_dc,
    }
    ats = ATS.create(
        model.n_ats, params, N_pre_diag=model.n_pre_diag, use_linear=False
    )
    evals, evecs = ats.eig_systems["vals"], ats.eig_systems["vecs"]
    evecs = np.asarray(evecs[:, : model.n_ats])
    phi = ats.linear_ops["phi"]
    h_without_nonlinear = ats.get_H_full() - ATS.get_H_nonlinear_static(
        phi,
        params["Ej"],
        params["dEj"],
        params["Ej2"],
        model.phi_sum_dc,
        model.phi_delta_dc,
    )

    omega = twopi * 6.4
    amplitude = 0.02125  # normalized Phi_Sigma/Phi0
    t = 0.37 * (twopi / omega)
    phi_sum_t = model.phi_sum_dc + amplitude * np.cos(omega * t)
    direct = h_without_nonlinear + ATS.get_H_nonlinear_static(
        phi,
        params["Ej"],
        params["dEj"],
        params["Ej2"],
        phi_sum_t,
        model.phi_delta_dc,
    )
    direct_projected = evecs.conj().T @ np.asarray(direct.data) @ evecs
    direct_projected -= float(evals[0]) * np.eye(model.n_ats)

    actual = model.hamiltonian((omega, amplitude))(t).full()
    assert np.allclose(actual, direct_projected, atol=2e-11)


def test_existing_qutip_floquet_solver_accepts_exact_ats_model():
    model = make_model(n_ats=8, n_pre_diag=40)
    analysis = FloquetAnalysis(
        model, state_indices=[0], options=Options(nsteps=4000, num_cpus=1)
    )
    modes, quasienergies = analysis.run_one_floquet(
        (2 * np.pi * 6.4, 0.02125)
    )
    assert len(modes) == model.n_ats
    assert len(quasienergies) == model.n_ats
    assert np.allclose([mode.norm() for mode in modes], 1.0)
