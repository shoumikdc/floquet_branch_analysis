"""Build the self-contained, executable FRARF ATS/full-RAR MIST notebook."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/examples/TwoQG_test_fluxonium.ipynb"

def md(text):
    return nbf.v4.new_markdown_cell(text.strip())

def code(text):
    return nbf.v4.new_code_cell(text.strip())

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "gkp2", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
nb["cells"] = [
md(r"""
# FRARF ATS and full-RAR MIST using the trusted Floquet package

This is the complete calculation: it constructs the isolated ATS and the full
storage--ATS--storage (RAR) Hamiltonian, calls the existing
`floquet_branch_analysis.FloquetAnalysis`, derives separately labelled diagnostics,
and makes the figures. It does **not** load hidden precomputed arrays.

The package's Floquet solver, smooth displaced-state fit, overlap definition,
Hungarian assignment, and branch tracking are unchanged. The only new package code
is `floquet/ats.py`, a Hamiltonian-only adapter for exact sum-flux driving.

Drive convention:
`phi_sum(t) = phi_sum_dc + delta_phi_sigma_peak_phi0*cos(omega_d*t)`, with
`phi_sum = Phi_Sigma/Phi0` and `Phi_Sigma=(Phi_L+Phi_R)/2`. Differential flux
is fixed at zero. Legacy `phi_ac=0.0425` corresponds to peak
`deltaPhi_Sigma/Phi0=0.02125` because `epsilon_0=pi*phi_ac=2*pi*deltaPhi_Sigma/Phi0`.
"""),
code("""
from pathlib import Path
import hashlib, sys, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt
from scipy.optimize import linear_sum_assignment

HERE = Path.cwd().resolve()
FLOQUET_REPO = Path('/Users/shoumikdc/crostini/weissharshfloquet/floquet_branch_analysis')
REPO = next((p for p in (HERE, *HERE.parents) if (p/'floquet/floquet.py').exists()),
            FLOQUET_REPO)
sys.path.insert(0, str(REPO))
FRARF = Path('/Users/shoumikdc/crostini/EQuS/gkp-fluxonium/bosonic-sims/project-gkp-2qg/devices/frarf')
sys.path.insert(0, str(FRARF))

from jaxquantum.core.settings import SETTINGS
SETTINGS['auto_tidyup_atol'] = 0.0
import jaxquantum as jqt
import jaxquantum.devices as qs
import floquet as ft
from rar_helpers import get_system_normal_rar

OUT = REPO/'docs/examples/out/frarf_full_rar_mist'
OUT.mkdir(parents=True, exist_ok=True)
print('repository:', REPO)
print('output:', OUT)
print('floquet_branch_analysis:', ft.__version__, 'QuTiP:', qt.__version__)
"""),
md("""
## Provenance

These hashes record the exact implementations used. The first three files are the
authors' solver/overlap/base-model code and are not edited by this work. `ats.py` is
the new Hamiltonian adapter; `rar_helpers.py` is the existing FRARF construction.
"""),
code("""
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

provenance_files = [
    REPO/'floquet/floquet.py', REPO/'floquet/displaced_state.py',
    REPO/'floquet/model.py', REPO/'floquet/ats.py', FRARF/'rar_helpers.py',
]
pd.DataFrame([{'file': str(p), 'sha256': sha256(p)} for p in provenance_files])
"""),
md(r"""
## Fixed comparison

Only $E_J/h$ changes: 16 GHz (legacy) versus 9 GHz (candidate). The calculation
holds $E_C/h=0.1742$ GHz, $E_L/h=21.07$ GHz, both direct couplings at 70 MHz,
storage targets at 3.0/3.4 GHz, dc sum flux at $0.25\Phi_0$, and dc differential
flux at zero. Thus no change to $E_C$, $E_L$, or coupling is assumed here.

At the exact symmetric sweet spot with $dE_J=0$, the static $E_J$ term vanishes.
The isolated ATS frequency is therefore unchanged when only $E_J$ changes, although
$E_J$ still controls the driven nonlinear term.
"""),
code("""
TWOPI = 2*np.pi
EC_GHZ, EL_GHZ = 0.1742, 21.07
G_A_GHZ = G_B_GHZ = 0.070
STORAGE_A_GHZ, STORAGE_B_GHZ = 3.0, 3.4
PHI_SUM_DC, PHI_DELTA_DC = 0.25, 0.0
LEGACY_PHI_SIGMA_PEAK = 0.02125
MATCHED_EJ9_PEAK = np.arcsin((16/9)*np.sin(TWOPI*LEGACY_PHI_SIGMA_PEAK))/TWOPI
print('legacy peak deltaPhi_Sigma/Phi0:', LEGACY_PHI_SIGMA_PEAK)
print('EJ=9 leading-drive match:', MATCHED_EJ9_PEAK, 'Phi0')
print('matched legacy phi_ac:', 2*MATCHED_EJ9_PEAK)
"""),
md("""
## 1. Isolated ATS, built and run here

`ATSModel` retains the full trigonometric sum-flux dependence. JAXQuantum builds
and projects the static operators; the trusted package performs the Floquet solve
and all overlap analysis.
"""),
code("""
def make_isolated_ats(ej_ghz, frequencies_ghz, amplitudes_phi0, dej_ghz=0.0):
    return ft.ATSModel(
        ec_ghz=EC_GHZ, el_ghz=EL_GHZ, ej_ghz=ej_ghz,
        dej_ghz=dej_ghz, ej2_ghz=0.0,
        phi_sum_dc=PHI_SUM_DC, phi_delta_dc=PHI_DELTA_DC,
        omega_d_values=TWOPI*np.asarray(frequencies_ghz),
        drive_amplitudes=np.asarray(amplitudes_phi0), n_ats=16, n_pre_diag=80,
    )

for ej in (9.0, 16.0):
    ats = make_isolated_ats(ej, [0.4, 6.4], [0.0, 0.052])
    print(f'EJ/h={ej:4.1f} GHz: isolated ATS f01={ats.H0.diag()[1]/TWOPI:.9f} GHz')
"""),
code("""
# Matrix-level verification against direct JAXQuantum at nonzero drive/time.
model_check = make_isolated_ats(9.0, [6.4], [LEGACY_PHI_SIGMA_PEAK])
p = dict(Ec=TWOPI*EC_GHZ, El=TWOPI*EL_GHZ, Ej=TWOPI*9.0,
         dEj=0.0, Ej2=0.0, phi_sum_ext=PHI_SUM_DC, phi_delta_ext=PHI_DELTA_DC)
ats_check = qs.ATS.create(16, p, N_pre_diag=80, use_linear=False)
evals, evecs = ats_check.eig_systems['vals'], np.asarray(ats_check.eig_systems['vecs'][:, :16])
phi = ats_check.linear_ops['phi']
h_without_nl = ats_check.get_H_full()-qs.ATS.get_H_nonlinear_static(phi, p['Ej'], 0, 0, PHI_SUM_DC, PHI_DELTA_DC)
omega = TWOPI*6.4
t_check = 0.37*(TWOPI/omega)
phi_sum_t = PHI_SUM_DC+LEGACY_PHI_SIGMA_PEAK*np.cos(omega*t_check)
direct = h_without_nl+qs.ATS.get_H_nonlinear_static(phi, p['Ej'], 0, 0, phi_sum_t, PHI_DELTA_DC)
direct_projected = evecs.conj().T@np.asarray(direct.data)@evecs-float(evals[0])*np.eye(16)
adapter_matrix = model_check.hamiltonian((omega, LEGACY_PHI_SIGMA_PEAK))(t_check).full()
print('isolated max |H_adapter-H_direct| =', np.max(np.abs(adapter_matrix-direct_projected)), 'rad/ns')
"""),
code("""
def run_isolated_ats(ej_ghz, window, nfreq=9, namp=13, save=True):
    center, span = (0.4, 0.20) if window == 'bs' else (6.4, 0.60)
    frequencies = np.linspace(center-span/2, center+span/2, nfreq)
    amplitudes = np.unique(np.r_[np.linspace(0, 0.10, namp), LEGACY_PHI_SIGMA_PEAK, MATCHED_EJ9_PEAK])
    model = make_isolated_ats(ej_ghz, frequencies, amplitudes)
    options = ft.Options(fit_range_fraction=.5, floquet_sampling_time_fraction=0,
                         fit_cutoff=3, overlap_cutoff=.8, nsteps=30000,
                         num_cpus=1, save_floquet_modes=True)
    analysis = ft.FloquetAnalysis(model, state_indices=[0], options=options)
    path = OUT/f'isolated_ats_ej{ej_ghz:g}_{window}_{nfreq}x{len(amplitudes)}.h5py'
    data = analysis.run(filepath=path if save else None)
    # Exactly the authors' scar definition; no alternate overlap calculation.
    p_mist = np.clip(1-data['displaced_state_overlaps'][:, :, 0]**2, 0, 1)
    outside_ground_t0 = np.clip(1-np.abs(data['floquet_modes'][:, :, 0, 0])**2, 0, 1)
    return dict(analysis=analysis, data=data, frequencies=frequencies,
                amplitudes=amplitudes, p_mist=p_mist,
                outside_ground_t0=outside_ground_t0, path=path)

isolated_control = run_isolated_ats(9.0, 'tms')
print("isolated control max authors' P_MIST:", isolated_control['p_mist'].max())
"""),
md("""
## 2. Full RAR construction, written out in this notebook

The code below calls the existing `get_system_normal_rar`, fully diagonalizes the
requested truncated RAR Hamiltonian, labels that *static* eigenbasis once by product
overlap, rotates the exact ATS phase operators into it, and supplies the resulting
Hamiltonian to the package. The one-time static labelling is not Floquet tracking;
the package still performs all Floquet branch tracking and overlap calculations.
"""),
code("""
def product_index(index, dims):
    return int(np.ravel_multi_index(index, dims))

def build_full_rar(ej_ghz, window, nfreq=21, namp=31, dej_ghz=0.0,
                   truncated_storage=2, truncated_ats=6,
                   normal_storage=12, normal_ats=80):
    n_cons = {
        'resonator_a': {'bare': 5, 'normal': normal_storage, 'truncated': truncated_storage},
        'ats': {'bare': 5, 'normal': normal_ats, 'truncated': truncated_ats},
        'resonator_b': {'bare': 5, 'normal': normal_storage, 'truncated': truncated_storage},
    }
    params = {
        'ResonatorA_frequency': TWOPI*STORAGE_A_GHZ,
        'ResonatorB_frequency': TWOPI*STORAGE_B_GHZ,
        'ATS__E_C': TWOPI*EC_GHZ, 'ATS__E_L': TWOPI*EL_GHZ,
        'ATS__E_J': TWOPI*ej_ghz, 'ATS__dE_J': TWOPI*dej_ghz,
        'ATS__E_J_2': 0.0, 'ATS__phi_sum_ext': PHI_SUM_DC,
        'ATS__phi_delta_ext': PHI_DELTA_DC,
        'g_ac': TWOPI*G_A_GHZ, 'g_cb': TWOPI*G_B_GHZ, 'N_CONS': n_cons,
    }
    system, participation, linear_metrics, linear_system = get_system_normal_rar(params)
    dims = tuple(system.Ns)
    h_static_product = np.asarray(system.get_H().data, dtype=complex)
    evals, evecs = np.linalg.eigh(h_static_product)

    # Static basis labels only. Package Floquet assignment is untouched.
    product_rows, eigen_cols = linear_sum_assignment(-np.abs(evecs)**2)
    eigen_for_product = np.empty(len(evals), dtype=int)
    eigen_for_product[product_rows] = eigen_cols
    transform = evecs[:, eigen_for_product]
    assigned_energies = evals[eigen_for_product]
    assignment_overlap = np.abs(transform[np.arange(len(evals)), np.arange(len(evals))])**2
    ground_energy = float(evals[0])
    h0 = np.diag(assigned_energies-ground_energy)

    phi = system.params['phi']
    cos_phi_product = np.asarray(jqt.cosm(phi).data, dtype=complex)
    sin_phi_product = np.asarray(jqt.sinm(phi).data, dtype=complex)
    cos_2phi_product = cos_phi_product@cos_phi_product-sin_phi_product@sin_phi_product
    sin_2phi_product = 2*cos_phi_product@sin_phi_product
    rotate = lambda op: transform.conj().T@op@transform

    labels = {s: product_index(idx, dims) for s, idx in {
        '000': (0,0,0), '100': (1,0,0), '001': (0,0,1), '101': (1,0,1)}.items()}
    targets = [labels['100'], labels['001']] if window == 'bs' else [labels['000'], labels['101']]
    target_labels = ['100', '001'] if window == 'bs' else ['000', '101']
    centers = {
        'bs': abs(assigned_energies[labels['100']]-assigned_energies[labels['001']])/TWOPI,
        'tms': abs(assigned_energies[labels['101']]-assigned_energies[labels['000']])/TWOPI,
    }
    center = float(centers[window])
    span = .08 if window == 'bs' else .16
    frequencies = np.linspace(center-span/2, center+span/2, nfreq)
    amplitudes = np.unique(np.r_[np.linspace(0, .10, namp), LEGACY_PHI_SIGMA_PEAK, MATCHED_EJ9_PEAK])

    model = ft.ExactSumFluxModel(
        H0=qt.Qobj(h0), cos_phi=qt.Qobj(rotate(cos_phi_product)),
        sin_phi=qt.Qobj(rotate(sin_phi_product)),
        cos_2phi=qt.Qobj(rotate(cos_2phi_product)),
        sin_2phi=qt.Qobj(rotate(sin_2phi_product)),
        ej_rad_per_ns=TWOPI*ej_ghz, dej_rad_per_ns=TWOPI*dej_ghz,
        ej2_rad_per_ns=0.0, phi_sum_dc=PHI_SUM_DC,
        phi_delta_dc=PHI_DELTA_DC, omega_d_values=TWOPI*frequencies,
        drive_amplitudes=amplitudes)

    # Matrix-level check against direct full-RAR JAXQuantum H(t).
    check_frequency, check_amplitude = TWOPI*center, LEGACY_PHI_SIGMA_PEAK
    check_time = .37*(TWOPI/check_frequency)
    check_flux = PHI_SUM_DC+check_amplitude*np.cos(check_frequency*check_time)
    nonlinear_dc = qs.ATS.get_H_nonlinear_static(phi, TWOPI*ej_ghz,
                                                  TWOPI*dej_ghz, 0, PHI_SUM_DC, PHI_DELTA_DC)
    direct = system.get_H()-nonlinear_dc+qs.ATS.get_H_nonlinear_static(
        phi, TWOPI*ej_ghz, TWOPI*dej_ghz, 0, check_flux, PHI_DELTA_DC)
    direct_rotated = rotate(np.asarray(direct.data))-ground_energy*np.eye(len(evals))
    adapter = model.hamiltonian((check_frequency, check_amplitude))(check_time).full()
    matrix_error = float(np.max(np.abs(direct_rotated-adapter)))
    ats_n_product = np.asarray((system.params['c'].dag()@system.params['c']).data)
    meta = dict(params=params, dims=dims, labels=labels, targets=targets,
                target_labels=target_labels, centers=centers, center=center,
                assignment_overlap=assignment_overlap, matrix_error=matrix_error,
                ats_number_eigenbasis=rotate(ats_n_product))
    return model, frequencies, amplitudes, meta
"""),
code("""
def run_full_rar(ej_ghz, window, nfreq=9, namp=13,
                 truncated_storage=2, truncated_ats=6, save=True):
    model, frequencies, amplitudes, meta = build_full_rar(
        ej_ghz, window, nfreq=nfreq, namp=namp,
        truncated_storage=truncated_storage, truncated_ats=truncated_ats)
    options = ft.Options(fit_range_fraction=.5, floquet_sampling_time_fraction=0,
                         fit_cutoff=2, overlap_cutoff=.8, nsteps=30000,
                         num_cpus=1, save_floquet_modes=True)
    analysis = ft.FloquetAnalysis(model, state_indices=meta['targets'], options=options)
    dims_tag = 'x'.join(map(str, meta['dims']))
    path = OUT/f'full_rar_ej{ej_ghz:g}_{window}_dims{dims_tag}_{nfreq}x{len(amplitudes)}.h5py'
    start = time.time()
    data = analysis.run(filepath=path if save else None)

    # Exact authors' metric. It can be large for desired target-target mixing.
    p_mist = np.clip(1-data['displaced_state_overlaps']**2, 0, 1)
    # Separate t=0 projection of those same authors-selected Floquet modes.
    modes = data['floquet_modes']
    target_weight = np.sum(np.abs(modes[:, :, :, meta['targets']])**2, axis=-1)
    outside_target_t0 = np.clip(1-target_weight, 0, 1)
    dims = meta['dims']
    ats_ground_indices = np.array([
        product_index((storage_a, 0, storage_b), dims)
        for storage_a in range(dims[0]) for storage_b in range(dims[2])
    ])
    ats_ground_weight = np.sum(np.abs(modes[:, :, :, ats_ground_indices])**2, axis=-1)
    ats_excited_weight_t0 = np.clip(1-ats_ground_weight, 0, 1)
    ats_occupation_t0 = np.real(np.einsum(
        '...i,ij,...j->...', modes.conj(), meta['ats_number_eigenbasis'], modes))
    return dict(analysis=analysis, data=data, frequencies=frequencies,
                amplitudes=amplitudes, meta=meta, p_mist=p_mist,
                outside_target_t0=outside_target_t0,
                ats_excited_weight_t0=ats_excited_weight_t0,
                ats_occupation_t0=ats_occupation_t0,
                path=path, elapsed_s=time.time()-start)

def plot_full_rar(result, vmax_outside=.05):
    f, a, meta = result['frequencies'], result['amplitudes'], result['meta']
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, label in enumerate(meta['target_labels']):
        entries = ((result['p_mist'][:, :, row], f"{label}: authors' P_MIST", .5),
                   (result['ats_excited_weight_t0'][:, :, row], f'{label}: ATS-excited weight at t=0', vmax_outside),
                   (result['ats_occupation_t0'][:, :, row], f'{label}: ATS occupation', None))
        for col, (values, title, vmax) in enumerate(entries):
            ax = axes[row, col]
            image = ax.pcolormesh(f, a, values.T, cmap='Reds', vmin=0,
                                  vmax=vmax, shading='nearest', rasterized=True)
            ax.axvline(meta['center'], color='.35', ls='--', lw=1)
            ax.set_title(title); fig.colorbar(image, ax=ax)
        axes[row,0].set_ylabel('peak deltaPhi_Sigma / Phi0')
    for ax in axes[-1]: ax.set_xlabel('drive frequency (GHz)')
    fig.suptitle(f"full RAR, EJ/h={meta['params']['ATS__E_J']/TWOPI:g} GHz, dims={meta['dims']}")
    return fig, axes
"""),
md(r"""
## 3. Live full-RAR calculation

This cell generates a fresh map from the code above; it does not load HDF5. The
authors' $P_\mathrm{MIST}$ can approach 0.5 at the BS avoided crossing because the
two desired target branches intentionally become 50/50 superpositions. The middle
column instead projects the same Floquet modes outside that two-state manifold.
"""),
code("""
pilot = run_full_rar(9.0, 'bs', nfreq=9, namp=13,
                     truncated_storage=2, truncated_ats=6)
print('full-RAR dimensions:', pilot['meta']['dims'])
print('static centers:', pilot['meta']['centers'])
print('minimum static label overlap:', pilot['meta']['assignment_overlap'].min())
print('max |H_adapter-H_direct|:', pilot['meta']['matrix_error'], 'rad/ns')
print('elapsed:', pilot['elapsed_s'], 's; saved:', pilot['path'])
pilot_fig, _ = plot_full_rar(pilot, vmax_outside=.20)
pilot_fig.savefig(OUT/'full_rar_ej9_bs_pilot.png', dpi=220)
plt.show()
"""),
md(r"""
## 4. Resolved maps and truncation convergence

Set `RUN_RESOLVED=True` to regenerate all four 21-by-33 maps: EJ 9/16 and BS/TMS.
The amplitude axis includes both the exact legacy normalized Phi_Sigma amplitude and the exact
leading-drive-matched EJ=9 amplitude.

A $2\times6\times2$ basis contains the target gate states but cannot represent
two-photon storage leakage. The example $3\times6\times3$ call below is therefore a
required convergence check before treating a sub-percent projection as a design
constraint.

The table reports $\pi/$quasienergy-gap as an interaction timescale. For BS this is
the swap-time estimate of the closed one-excitation doublet. For TMS it is only the
local $|000\rangle\leftrightarrow|101\rangle$ coupling timescale, **not** a complete
TMS gate time, because the resonant squeezing ladder continues through
$|202\rangle,|303\rangle,\ldots$.
"""),
code("""
RUN_RESOLVED = True
resolved = {}
if RUN_RESOLVED:
    for ej in (9.0, 16.0):
        for window in ('bs', 'tms'):
            print('running', ej, window)
            result = run_full_rar(ej, window, nfreq=21, namp=31,
                                  truncated_storage=2, truncated_ats=6)
            resolved[(ej, window)] = result
            print('elapsed', result['elapsed_s'], 's; saved', result['path'])
            figure, _ = plot_full_rar(result)
            figure.savefig(OUT/f'full_rar_ej{ej:g}_{window}_resolved.png', dpi=220)
            plt.show()

def summarize_operating_point(result, amplitude, label):
    ai = int(np.argmin(abs(result['amplitudes']-amplitude)))
    omega = TWOPI*result['frequencies']
    targets = result['meta']['targets']
    qe = result['data']['quasienergies']
    delta = qe[:, ai, targets[0]]-qe[:, ai, targets[1]]
    gap = np.abs((delta+omega/2) % omega-omega/2)
    fi = int(np.argmin(gap))
    return dict(
        EJ_GHz=result['meta']['params']['ATS__E_J']/TWOPI,
        tone='BS' if result['meta']['target_labels']==['100','001'] else 'TMS',
        amplitude_label=label,
        phi_sigma_peak=result['amplitudes'][ai],
        resonant_frequency_GHz=result['frequencies'][fi],
        quasienergy_gap_MHz=gap[fi]/TWOPI*1e3,
        pi_over_gap_timescale_us=np.pi/gap[fi]/1e3,
        max_ATS_excited_weight_at_resonance=result['ats_excited_weight_t0'][fi, ai].max(),
        max_ATS_excited_weight_across_window=result['ats_excited_weight_t0'][:, ai].max(),
        max_outside_target_at_resonance=result['outside_target_t0'][fi, ai].max(),
        max_outside_target_across_window=result['outside_target_t0'][:, ai].max(),
    )

if resolved:
    operating_rows = []
    for key, result in resolved.items():
        for label, amplitude in (
            ('legacy', LEGACY_PHI_SIGMA_PEAK),
            ('EJ9 leading-drive match', MATCHED_EJ9_PEAK),
            ('0.052 Phi0', 0.052),
        ):
            operating_rows.append(summarize_operating_point(result, amplitude, label))
    operating_summary = pd.DataFrame(operating_rows)
    display(operating_summary)

RUN_CONVERGENCE = True
converged = {}
if RUN_CONVERGENCE:
    for window in ('bs', 'tms'):
        print('running higher-storage truncation: EJ=9', window)
        result = run_full_rar(9.0, window, nfreq=3, namp=31,
                              truncated_storage=3, truncated_ats=6)
        converged[window] = result
        print('elapsed', result['elapsed_s'], 's; saved', result['path'])
        figure, _ = plot_full_rar(result)
        figure.savefig(OUT/f'full_rar_ej9_{window}_dims3x6x3_convergence.png', dpi=220)
        plt.show()

if converged:
    convergence_rows = []
    for window, result in converged.items():
        for label, amplitude in (
            ('EJ9 leading-drive match', MATCHED_EJ9_PEAK),
            ('0.052 Phi0', 0.052),
        ):
            convergence_rows.append(summarize_operating_point(result, amplitude, label))
    convergence_summary = pd.DataFrame(convergence_rows)
    display(convergence_summary)
"""),
md(r"""
## Interpretation contract

1. **Authors' $P_\mathrm{MIST}=1-|\langle u|u_\mathrm{smooth}\rangle|^2$** is used
   without alteration. At a desired gate avoided crossing it can be large because
   the individual target branches mix.
2. **ATS-excited weight at Floquet phase $t=0$** sums over every retained storage
   state and asks only whether the ATS is outside its static ground sector. This is
   the central ATS-leakage proxy, but it is phase dependent and includes reversible
   dressing.
3. **Outside the target two-state manifold at $t=0$** is retained in the tables.
   For TMS it includes the intended squeezing ladder (for example $|202\rangle$),
   so it must not be labelled ATS leakage.
4. **ATS occupation at $t=0$** is a related diagnostic that weights higher ATS
   levels by excitation number.

A gate fidelity/leakage claim still requires a truncation-converged basis plus
shaped-pulse time evolution. This notebook does not equate an isolated-ATS scar,
target-state hybridization, or a single-phase projection with 99% gate fidelity.
For TMS, direct time evolution is additionally necessary because a two-state Floquet
gap does not converge to the dynamics of the full resonant squeezing ladder.
"""),
]

nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
