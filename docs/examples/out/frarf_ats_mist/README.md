# FRARF isolated-ATS trusted Floquet cross-check

## Trust boundary

These results use the existing `floquet_branch_analysis.FloquetAnalysis` without
changes to its QuTiP Floquet solver, displaced-state fit, overlap calculation,
Hungarian assignment, or Blais branch tracking. The only new solver input is
`floquet.ATSModel`, which supplies the exact periodically sum-flux-driven ATS
Hamiltonian.

The adapter Hamiltonian was compared at nonzero drive and an arbitrary time to a
direct JAXQuantum ATS matrix projected into the same eigenbasis. The maximum matrix
difference was `1.25e-12 rad/ns`.

## Fixed comparison

- `EC/h = 0.1742 GHz`
- `EL/h = 21.07 GHz`
- `EJ/h = 9` versus `16 GHz`
- `phi_sum_dc = 0.25 Phi0`
- `phi_delta_dc = 0`
- sum-flux drive only
- physical peak sum-flux amplitude from `0` through `0.06 Phi0`
- 16 retained ATS levels after diagonalization from an 80-state oscillator basis

Both values of EJ give `f01 = 5.418777722 GHz` at this exact symmetric sweet spot
when EC and EL are held fixed.

## Metrics

- `P_MIST = 1 - displaced_state_overlaps**2` is exactly the package authors'
  Xiao/Venkatraman scar metric.
- `tracked_mean_excitation` is exactly the package authors' Blais/Hungarian
  amplitude-step branch result.
- `outside_bare_ground` is an additional diagnostic computed from the saved Floquet
  mode: `1 - abs(floquet_mode[ground_component])**2`. It includes ordinary reversible
  dressing and is not automatically irreversible leakage.
- The package array called `bare_state_overlaps` is not relabeled as bare-ground
  leakage here. After the first iterative fit segment, its reference is the previous
  fitted dressed state by design.

## Main result at 0.052 Phi0 physical peak

| EJ/h | tone | max P_MIST in window | P_MIST at tone | outside undriven ground | tracked mean excitation |
|---:|---:|---:|---:|---:|---:|
| 9 GHz | 0.4 GHz BS | 3.91e-9 | 1.01e-10 | 2.712e-3 | 5.433e-3 |
| 9 GHz | 6.4 GHz TMS | 2.46e-4 | 1.56e-8 | 3.144e-3 | 6.298e-3 |
| 16 GHz | 0.4 GHz BS | 7.44e-7 | 2.36e-8 | 1.107e-2 | 2.234e-2 |
| 16 GHz | 6.4 GHz TMS | 2.18e-4 | 7.17e-8 | 6.816e-3 | 1.367e-2 |

At `dEJ/EJ = 3%` for the 9 GHz case, the maximum scar remains `4.98e-9`
in the BS window and `1.55e-5` in the TMS window. The tracked mean excitations
at the operating tones and `0.052 Phi0` are `0.00550` and `0.00675`.

The broad 1--12 GHz control scan shows the expected strong even-parity resonance
near `f02 = 10.84 GHz` and multiphoton features near 5.4 and 3.6 GHz. This verifies
that the clean 0.4 and 6.4 GHz windows are not caused by a broken scar pipeline.

## Pump comparison

The legacy `phi_ac=0.0425` corresponds to a physical peak sum-flux modulation of
`0.02125 Phi0`. Holding EC, EL, and storage couplings fixed, matching the leading
16 GHz drive term with EJ/h=9 GHz requires `0.0380264 Phi0` physical peak
(`legacy phi_ac=0.0760528`). At those matched-drive points, isolated-ATS dressing
is essentially identical. A `0.052 Phi0` pump is 1.356 times the old leading drive.

## What this does not prove

This is not full-RAR gate leakage. Storage coupling adds dressed branches; the old
time-domain gate simulation also includes pulse turn-on, off-resonant coherent
population, and residual population at the stopping time. The next calculation must
hierarchically diagonalize the full RAR, rotate the exact sum-flux operators into its
eigenbasis, and pass that periodic Hamiltonian through the same unchanged
`FloquetAnalysis`.
