"""Run isolated-FRARF-ATS MIST maps with floquet_branch_analysis.

This script intentionally uses the package's existing ``FloquetAnalysis`` for
QuTiP Floquet modes, Xiao/Venkatraman displaced-state fits, and Blais branch
tracking.  ``ATSModel`` supplies only the exact periodic ATS Hamiltonian.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import floquet as ft


WINDOWS_GHZ = {
    "bs": (0.30, 0.50, 0.40),
    "tms": (6.10, 6.70, 6.40),
    "wide": (1.00, 12.00, 6.40),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ej", type=float, required=True, help="EJ/h in GHz")
    parser.add_argument("--ec", type=float, required=True, help="EC/h in GHz")
    parser.add_argument("--el", type=float, required=True, help="EL/h in GHz")
    parser.add_argument("--window", choices=WINDOWS_GHZ, required=True)
    parser.add_argument("--dej", type=float, default=0.0, help="dEJ/h in GHz")
    parser.add_argument("--phi-sum-dc", type=float, default=0.25)
    parser.add_argument("--phi-delta-dc", type=float, default=0.0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--nfreq", type=int, default=31)
    parser.add_argument("--namp", type=int, default=31)
    parser.add_argument("--max-phi-sigma", type=float, default=0.10)
    parser.add_argument("--n-ats", type=int, default=16)
    parser.add_argument("--n-pre-diag", type=int, default=80)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fmin, fmax, operating_frequency = WINDOWS_GHZ[args.window]
    frequencies_ghz = np.linspace(fmin, fmax, args.nfreq)
    amplitudes_phi_sigma = np.linspace(0.0, args.max_phi_sigma, args.namp)

    model = ft.ATSModel(
        ec_ghz=args.ec,
        el_ghz=args.el,
        ej_ghz=args.ej,
        dej_ghz=args.dej,
        ej2_ghz=0.0,
        phi_sum_dc=args.phi_sum_dc,
        phi_delta_dc=args.phi_delta_dc,
        omega_d_values=2 * np.pi * frequencies_ghz,
        drive_amplitudes=amplitudes_phi_sigma,
        n_ats=args.n_ats,
        n_pre_diag=args.n_pre_diag,
    )
    options = ft.Options(
        fit_range_fraction=0.5,
        floquet_sampling_time_fraction=0.0,
        fit_cutoff=3,
        overlap_cutoff=0.8,
        nsteps=args.nsteps,
        num_cpus=args.num_cpus,
        save_floquet_modes=True,
    )
    analysis = ft.FloquetAnalysis(model, state_indices=[0], options=options)
    prefix = f"{args.tag}_" if args.tag else ""
    stem = f"{prefix}ats_ej{args.ej:g}_dej{args.dej:g}_{args.window}_{args.nfreq}x{args.namp}"
    h5_path = args.out / f"{stem}.h5py"
    data = analysis.run(filepath=h5_path)

    p_mist = np.clip(1.0 - data["displaced_state_overlaps"][:, :, 0] ** 2, 0.0, 1.0)
    p_bare = np.clip(
        1.0 - np.abs(data["floquet_modes"][:, :, 0, 0]) ** 2, 0.0, 1.0
    )
    p_tracking_reference = np.clip(
        1.0 - data["bare_state_overlaps"][:, :, 0] ** 2, 0.0, 1.0
    )
    branch_n = data["avg_excitation"][:, :, 0]

    csv_path = args.out / f"{stem}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "drive_frequency_ghz",
                "delta_phi_sigma_peak_phi0",
                "legacy_phi_ac",
                "p_mist_ground",
                "p_outside_bare_ground",
                "p_outside_iterative_tracking_reference",
                "tracked_ground_branch_mean_excitation",
            ]
        )
        for fi, frequency in enumerate(frequencies_ghz):
            for ai, amplitude in enumerate(amplitudes_phi_sigma):
                writer.writerow(
                    [
                        frequency,
                        amplitude,
                        2.0 * amplitude,
                        p_mist[fi, ai],
                        p_bare[fi, ai],
                        p_tracking_reference[fi, ai],
                        branch_n[fi, ai],
                    ]
                )

    metadata = {
        "solver": "floquet_branch_analysis.FloquetAnalysis -> qutip.FloquetBasis",
        "model": "floquet.ATSModel exact nonlinear sum-flux drive",
        "energies_ghz": {
            "EC": args.ec,
            "EL": args.el,
            "EJ": args.ej,
            "dEJ": args.dej,
            "EJ2": 0.0,
        },
        "dEJ_over_EJ": args.dej / args.ej,
        "bias_phi0": {"sum": args.phi_sum_dc, "differential": args.phi_delta_dc},
        "drive": "sum flux only; differential flux fixed",
        "amplitude_convention": {
            "delta_phi_sigma_peak_phi0": "peak Phi_Sigma/Phi0, Phi_Sigma=(Phi_L+Phi_R)/2",
            "legacy_phi_ac": "2 * delta_phi_sigma_peak_phi0",
        },
        "truncation": {"n_ats": args.n_ats, "n_pre_diag": args.n_pre_diag},
        "grid": {
            "window": args.window,
            "nfreq": args.nfreq,
            "namp": args.namp,
            "max_phi_sigma": args.max_phi_sigma,
        },
        "options": vars(options),
        "ats_f01_ghz": float(model.H0.diag()[1] / (2 * np.pi)),
        "metrics": {
            "p_mist_ground": "1 - |<Floquet ground branch|smooth fitted dressed ground>|^2",
            "p_outside_bare_ground": "1 - |<Floquet ground branch|undriven ground>|^2",
            "p_outside_iterative_tracking_reference": "internal overlap used to identify the branch; after the first fit segment its reference is the previous fitted dressed state, not the undriven ground",
        },
    }
    with (args.out / f"{stem}.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    extent = [fmin, fmax, 0.0, args.max_phi_sigma]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True, constrained_layout=True)
    plot_specs = [
        (p_mist.T, r"$P_{\rm MIST}^{(g)}$", 0.5),
        (p_bare.T, r"$1-|\langle g|u_g(0)\rangle|^2$", 1.0),
        (branch_n.T, r"tracked $\langle N_{\rm ATS}\rangle$", None),
    ]
    for ax, (values, title, vmax) in zip(axes, plot_specs, strict=True):
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            cmap="Reds",
            vmin=0.0,
            vmax=vmax,
        )
        ax.axvline(operating_frequency, color="0.35", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(r"drive frequency $f_d$ (GHz)")
        fig.colorbar(image, ax=ax)
    axes[0].set_ylabel(r"peak $\delta\Phi_\Sigma/\Phi_0$")
    fig.suptitle(
        rf"isolated ATS: $E_J/h={args.ej:g}$ GHz, $dE_J/h={args.dej:g}$ GHz; "
        rf"$f_{{01}}={metadata['ats_f01_ghz']:.4f}$ GHz"
    )
    figure_path = args.out / f"{stem}.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print(json.dumps({"h5": str(h5_path), "csv": str(csv_path), "figure": str(figure_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
