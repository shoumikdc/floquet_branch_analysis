"""Full-RAR MIST pilot using the unchanged floquet_branch_analysis algorithms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import curve_fit, linear_sum_assignment

import floquet as ft


DEFAULT_FRARF = Path(
    "/Users/shoumikdc/crostini/EQuS/gkp-fluxonium/bosonic-sims/"
    "project-gkp-2qg/devices/frarf"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ej", type=float, required=True)
    parser.add_argument("--ec", type=float, required=True)
    parser.add_argument("--el", type=float, required=True)
    parser.add_argument("--window", choices=("bs", "tms"), required=True)
    parser.add_argument("--dej", type=float, default=0.0)
    parser.add_argument("--storage-a", type=float, default=3.0)
    parser.add_argument("--storage-b", type=float, default=3.4)
    parser.add_argument("--g-a-mhz", type=float, default=70.0)
    parser.add_argument("--g-b-mhz", type=float, default=70.0)
    parser.add_argument("--phi-sum-dc", type=float, default=0.25)
    parser.add_argument("--phi-delta-dc", type=float, default=0.0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--nfreq", type=int, default=9)
    parser.add_argument("--namp", type=int, default=13)
    parser.add_argument("--max-phi-sigma", type=float, default=0.10)
    parser.add_argument("--span-ghz", type=float, default=None)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--truncated-storage", type=int, default=2)
    parser.add_argument("--truncated-ats", type=int, default=6)
    parser.add_argument("--normal-storage", type=int, default=12)
    parser.add_argument("--normal-ats", type=int, default=80)
    parser.add_argument("--frarf", type=Path, default=DEFAULT_FRARF)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def product_index(index: tuple[int, int, int], dims: tuple[int, int, int]) -> int:
    return int(np.ravel_multi_index(index, dims))


def build_model(args: argparse.Namespace):
    sys.path.insert(0, str(args.frarf))
    import jaxquantum as jqt
    import jaxquantum.devices as qs
    from jaxquantum.core.settings import SETTINGS
    from rar_helpers import get_system_normal_rar

    SETTINGS["auto_tidyup_atol"] = 0.0
    twopi = 2 * np.pi
    n_cons = {
        "resonator_a": {
            "bare": 5,
            "normal": args.normal_storage,
            "truncated": args.truncated_storage,
        },
        "ats": {
            "bare": 5,
            "normal": args.normal_ats,
            "truncated": args.truncated_ats,
        },
        "resonator_b": {
            "bare": 5,
            "normal": args.normal_storage,
            "truncated": args.truncated_storage,
        },
    }
    params = {
        "ResonatorA_frequency": twopi * args.storage_a,
        "ResonatorB_frequency": twopi * args.storage_b,
        "ATS__E_C": twopi * args.ec,
        "ATS__E_L": twopi * args.el,
        "ATS__E_J": twopi * args.ej,
        "ATS__dE_J": twopi * args.dej,
        "ATS__E_J_2": 0.0,
        "ATS__phi_sum_ext": args.phi_sum_dc,
        "ATS__phi_delta_ext": args.phi_delta_dc,
        "g_ac": twopi * args.g_a_mhz / 1000.0,
        "g_cb": twopi * args.g_b_mhz / 1000.0,
        "N_CONS": n_cons,
    }
    system, participation, linear_metrics, linear_system = get_system_normal_rar(params)
    dims = tuple(system.Ns)
    h_static_product = np.asarray(system.get_H().data, dtype=complex)
    evals, evecs = np.linalg.eigh(h_static_product)

    # Assign every static eigenvector to one product label once.  This is only a
    # basis-labeling step; Floquet branch assignment remains the package's own code.
    product_rows, eigen_cols = linear_sum_assignment(-np.abs(evecs) ** 2)
    eigen_for_product = np.empty(len(evals), dtype=int)
    eigen_for_product[product_rows] = eigen_cols
    transform = evecs[:, eigen_for_product]
    assigned_energies = evals[eigen_for_product]
    assignment_overlap = np.abs(transform[np.arange(len(evals)), np.arange(len(evals))]) ** 2
    ground_energy = float(evals[0])
    h0 = np.diag(assigned_energies - ground_energy)

    phi = system.params["phi"]
    cos_phi_product = np.asarray(jqt.cosm(phi).data, dtype=complex)
    sin_phi_product = np.asarray(jqt.sinm(phi).data, dtype=complex)
    cos_2phi_product = cos_phi_product @ cos_phi_product - sin_phi_product @ sin_phi_product
    sin_2phi_product = 2.0 * cos_phi_product @ sin_phi_product

    def rotate(operator: np.ndarray) -> np.ndarray:
        return transform.conj().T @ operator @ transform

    idx000 = product_index((0, 0, 0), dims)
    idx100 = product_index((1, 0, 0), dims)
    idx001 = product_index((0, 0, 1), dims)
    idx101 = product_index((1, 0, 1), dims)
    static_frequencies = {
        "bs": abs(assigned_energies[idx100] - assigned_energies[idx001]) / twopi,
        "tms": abs(assigned_energies[idx101] - assigned_energies[idx000]) / twopi,
    }
    center = float(static_frequencies[args.window])
    span = args.span_ghz if args.span_ghz is not None else (0.08 if args.window == "bs" else 0.16)
    frequencies_ghz = np.linspace(center - span / 2, center + span / 2, args.nfreq)
    amplitudes_phi_sigma = np.linspace(0.0, args.max_phi_sigma, args.namp)
    model = ft.ExactSumFluxModel(
        H0=qt.Qobj(h0),
        cos_phi=qt.Qobj(rotate(cos_phi_product)),
        sin_phi=qt.Qobj(rotate(sin_phi_product)),
        cos_2phi=qt.Qobj(rotate(cos_2phi_product)),
        sin_2phi=qt.Qobj(rotate(sin_2phi_product)),
        ej_rad_per_ns=twopi * args.ej,
        dej_rad_per_ns=twopi * args.dej,
        ej2_rad_per_ns=0.0,
        phi_sum_dc=args.phi_sum_dc,
        phi_delta_dc=args.phi_delta_dc,
        omega_d_values=twopi * frequencies_ghz,
        drive_amplitudes=amplitudes_phi_sigma,
    )
    targets = [idx100, idx001] if args.window == "bs" else [idx000, idx101]

    # Independent exact-H(t) cross-check against the direct product-basis model.
    check_frequency = twopi * center
    check_amplitude = min(0.02125, args.max_phi_sigma)
    check_time = 0.37 * twopi / check_frequency
    check_flux = args.phi_sum_dc + check_amplitude * np.cos(check_frequency * check_time)
    nonlinear_dc = qs.ATS.get_H_nonlinear_static(
        phi, twopi * args.ej, twopi * args.dej, 0.0,
        args.phi_sum_dc, args.phi_delta_dc,
    )
    direct = system.get_H() - nonlinear_dc + qs.ATS.get_H_nonlinear_static(
        phi, twopi * args.ej, twopi * args.dej, 0.0,
        check_flux, args.phi_delta_dc,
    )
    direct_rotated = rotate(np.asarray(direct.data, dtype=complex)) - ground_energy * np.eye(len(evals))
    adapter = model.hamiltonian((check_frequency, check_amplitude))(check_time).full()
    matrix_error = float(np.max(np.abs(direct_rotated - adapter)))

    ats_number_product = np.asarray(
        (system.params["c"].dag() @ system.params["c"]).data, dtype=complex
    )
    ats_ground_projector_product = np.zeros_like(h_static_product)
    for storage_a_index in range(dims[0]):
        for storage_b_index in range(dims[2]):
            index = product_index((storage_a_index, 0, storage_b_index), dims)
            ats_ground_projector_product[index, index] = 1.0
    metadata = {
        "params": params,
        "dims": dims,
        "targets": targets,
        "target_labels": ["100", "001"] if args.window == "bs" else ["000", "101"],
        "center_frequency_ghz": center,
        "static_frequencies_ghz": {key: float(value) for key, value in static_frequencies.items()},
        "minimum_product_assignment_overlap": float(assignment_overlap.min()),
        "target_assignment_overlaps": [float(assignment_overlap[index]) for index in targets],
        "exact_matrix_crosscheck_rad_per_ns": matrix_error,
        "ats_number_eigenbasis": rotate(ats_number_product),
        "ats_ground_projector_eigenbasis": rotate(ats_ground_projector_product),
    }
    return model, targets, frequencies_ghz, amplitudes_phi_sigma, metadata


def main() -> None:
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    model, targets, frequencies, amplitudes, metadata = build_model(args)
    options = ft.Options(
        fit_range_fraction=0.5,
        floquet_sampling_time_fraction=0.0,
        fit_cutoff=2,
        overlap_cutoff=0.8,
        nsteps=args.nsteps,
        num_cpus=args.num_cpus,
        save_floquet_modes=True,
    )
    analysis = ft.FloquetAnalysis(model, state_indices=targets, options=options)
    dims_tag = "x".join(str(value) for value in metadata["dims"])
    prefix = f"{args.tag}_" if args.tag else ""
    stem = (
        f"{prefix}full_rar_ej{args.ej:g}_dej{args.dej:g}_{args.window}_"
        f"dims{dims_tag}_{args.nfreq}x{args.namp}"
    )
    h5_path = args.out / f"{stem}.h5py"
    data = analysis.run(filepath=h5_path)

    p_mist = np.clip(1.0 - data["displaced_state_overlaps"] ** 2, 0.0, 1.0)
    modes = data["floquet_modes"]
    computational_weight = np.sum(np.abs(modes[:, :, :, targets]) ** 2, axis=-1)
    outside_computational = np.clip(1.0 - computational_weight, 0.0, 1.0)
    n_ats_op = metadata.pop("ats_number_eigenbasis")
    p_ats_ground_op = metadata.pop("ats_ground_projector_eigenbasis")
    ats_occupation = np.real(
        np.einsum("...i,ij,...j->...", modes.conj(), n_ats_op, modes)
    )
    ats_ground_weight = np.real(
        np.einsum("...i,ij,...j->...", modes.conj(), p_ats_ground_op, modes)
    )
    ats_excited_weight = np.clip(1.0 - ats_ground_weight, 0.0, 1.0)

    with (args.out / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "drive_frequency_ghz", "delta_phi_sigma_peak_phi0", "tracked_label",
                "p_mist_authors", "outside_target_two_state_manifold",
                "ats_excited_product_weight_t0", "ats_occupation_t0",
            ]
        )
        for fi, frequency in enumerate(frequencies):
            for ai, amplitude in enumerate(amplitudes):
                for ti, label in enumerate(metadata["target_labels"]):
                    writer.writerow(
                        [frequency, amplitude, label, p_mist[fi, ai, ti],
                         outside_computational[fi, ai, ti],
                         ats_excited_weight[fi, ai, ti], ats_occupation[fi, ai, ti]]
                    )

    # The Blais-tracked quasienergies retain all static branches.  Near a
    # one-pump-photon target crossing, the wrapped splitting follows
    # sqrt((f-f_res)^2 + gap^2).  Fitting that curve avoids mistaking the finite
    # frequency-grid spacing for the avoided-crossing gap.
    quasienergies = data["quasienergies"]
    omega_d = 2.0 * np.pi * frequencies
    target_delta = quasienergies[:, :, targets[0]] - quasienergies[:, :, targets[1]]
    wrapped_gap = np.abs(
        (target_delta + omega_d[:, None] / 2.0) % omega_d[:, None]
        - omega_d[:, None] / 2.0
    )
    best_frequency_indices = np.argmin(wrapped_gap, axis=0)
    summary_path = args.out / f"{stem}_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "delta_phi_sigma_peak_phi0", "fitted_resonance_frequency_ghz",
                "fitted_target_gap_mhz", "fitted_swap_time_us",
                "fit_rms_residual_mhz", "nearest_sample_frequency_ghz",
                "max_p_mist_authors_t0",
                "max_outside_target_t0", "max_ats_excited_product_weight_t0",
                "max_ats_occupation_t0",
            ]
        )
        for ai, amplitude in enumerate(amplitudes):
            fi_grid = int(best_frequency_indices[ai])
            lo = max(0, fi_grid - 2)
            hi = min(len(frequencies), fi_grid + 3)
            fit_frequencies = frequencies[lo:hi]
            fit_splittings = wrapped_gap[lo:hi, ai] / (2.0 * np.pi)

            def avoided_crossing(frequency, resonance, gap):
                return np.sqrt((frequency - resonance) ** 2 + gap ** 2)

            try:
                fit, _ = curve_fit(
                    avoided_crossing,
                    fit_frequencies,
                    fit_splittings,
                    p0=(frequencies[fi_grid], max(float(fit_splittings.min()), 1e-12)),
                    bounds=((frequencies[0], 0.0), (frequencies[-1], np.inf)),
                    maxfev=10000,
                )
                resonance, gap_ghz = (float(value) for value in fit)
                residual_mhz = float(
                    np.sqrt(np.mean((fit_splittings - avoided_crossing(fit_frequencies, *fit)) ** 2))
                    * 1000.0
                )
            except (RuntimeError, ValueError):
                resonance = float(frequencies[fi_grid])
                gap_ghz = float(fit_splittings.min())
                residual_mhz = np.nan
            fi = int(np.argmin(np.abs(frequencies - resonance)))
            swap_time_us = np.inf if gap_ghz == 0.0 else 1.0 / (2.0 * gap_ghz) / 1000.0
            writer.writerow(
                [
                    amplitude, resonance, gap_ghz * 1000.0, swap_time_us,
                    residual_mhz, frequencies[fi], np.max(p_mist[fi, ai]),
                    np.max(outside_computational[fi, ai]),
                    np.max(ats_excited_weight[fi, ai]),
                    np.max(ats_occupation[fi, ai]),
                ]
            )

    serializable_metadata = {
        key: value for key, value in metadata.items() if key != "params"
    }
    serializable_metadata.update(
        {
            "solver": "unchanged floquet_branch_analysis.FloquetAnalysis",
            "construction": "rar_helpers.get_system_normal_rar then full static eigendecomposition",
            "energies_ghz": {"EC": args.ec, "EL": args.el, "EJ": args.ej, "dEJ": args.dej},
            "dEJ_over_EJ": args.dej / args.ej,
            "couplings_mhz": {"gA": args.g_a_mhz, "gB": args.g_b_mhz},
            "storage_frequencies_ghz": {"A": args.storage_a, "B": args.storage_b},
            "bias_phi0": {"sum": args.phi_sum_dc, "differential": args.phi_delta_dc},
            "amplitude_convention": {
                "delta_phi_sigma_peak_phi0": "peak Phi_Sigma/Phi0, Phi_Sigma=(Phi_L+Phi_R)/2",
            },
            "grid": {"nfreq": args.nfreq, "namp": args.namp, "max_phi_sigma": args.max_phi_sigma},
            "max_p_mist": float(p_mist.max()),
            "max_outside_target_manifold": float(outside_computational.max()),
            "max_ats_excited_product_weight_t0": float(ats_excited_weight.max()),
            "max_ats_occupation": float(ats_occupation.max()),
            "summary_csv": str(summary_path),
            "metric_scope": "Floquet modes at t=0; not a within-period maximum or open-system heating rate",
        }
    )
    with (args.out / f"{stem}.json").open("w") as handle:
        json.dump(serializable_metadata, handle, indent=2)

    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True, sharey=True, constrained_layout=True)
    for row, label in enumerate(metadata["target_labels"]):
        for ax, values, title, vmax in (
            (axes[row, 0], p_mist[:, :, row], rf"{label}: package $P_{{MIST}}$", 0.5),
            (axes[row, 1], outside_computational[:, :, row], rf"{label}: outside target manifold", 0.1),
            (axes[row, 2], ats_excited_weight[:, :, row], rf"{label}: $1-P_{{c0}}$ at $t=0$", 0.1),
            (axes[row, 3], ats_occupation[:, :, row], rf"{label}: $\langle n_{{ATS}}\rangle$ at $t=0$", None),
        ):
            image = ax.pcolormesh(frequencies, amplitudes, values.T, cmap="Reds", vmin=0, vmax=vmax, shading="nearest")
            ax.axvline(metadata["center_frequency_ghz"], color="0.35", ls="--", lw=1)
            ax.set_title(title)
            fig.colorbar(image, ax=ax)
        axes[row, 0].set_ylabel(r"peak $\delta\Phi_\Sigma/\Phi_0$")
    for ax in axes[-1]:
        ax.set_xlabel("drive frequency (GHz)")
    fig.suptitle(
        rf"full RAR, $E_J/h={args.ej:g}$ GHz, {args.window.upper()}, dims={metadata['dims']}"
    )
    fig.savefig(args.out / f"{stem}.png", dpi=220)
    plt.close(fig)
    print(json.dumps(serializable_metadata, indent=2))


if __name__ == "__main__":
    main()
