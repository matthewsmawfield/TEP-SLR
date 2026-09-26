#!/usr/bin/env python3
"""
Step 5.2: VLBI-SLR scale-drift channel -- predicted amplitude and the
local drift-participation bound (Issue 8-7).

Under TEP the ambient conformal factor A(t) carries the cosmological drift
H_drift = -du/dt ~ H_0 (Paper 0, S8: the drift is carried by the
matter-hosted field values, which include the terrestrial environment).
Rule 8 makes a uniform rescaling invisible; an ITRF scale split between two
techniques is observable only because the frames anchor scale in different
metric sectors:

  * VLBI anchors scale geometrically: station positions are solved from
    quasar-fixed directions and group delays measured on matter clocks and
    converted through the defined speed of light. The realized length unit
    is the matter-metric meter, so s_VLBI ~ A(t) and s_dot/s = A_dot/A.

  * SLR anchors scale dynamically: station coordinates are solved so that
    measured ranges are consistent with an orbit integrated from the
    adopted gravitational parameter GM and a timescale supplied by the same
    drifting matter clocks. The GM anchor itself is drift-free (a fixed
    conventional constant of the g-sector solution), but the orbit's
    inferred size inherits a partial track of the drift through the
    measured period: a_inf = (GM T_tilde^2/4 pi^2)^(1/3) ~ A^(2/3) when the
    g-coordinate period T_g is read on drifting clocks, T_tilde = A T_g.
    Station radius ~ a_inf - rho_tilde with the range rho_tilde ~ A rho_g,
    so the SLR scale inherits an environment-dependent fraction f_dyn of
    the ambient drift rather than 0 or 1.

The predicted inter-frame split therefore lies between 0 (full internal
tracking) and A_dot/A (no tracking). The two limiting bookkeeping cases
computed here bracket the channel; the observed ~0.2 ppb (2013-2025,
ITRF2020: Altamimi et al. 2023; Kern et al. 2024; Hellmers et al. 2025)
bounds the locally realized participation fraction.

Outputs: results/outputs/step_5_2_vlbi_slr_scale_drift.json
"""

import json
import os

S_PER_YR = 3.15576e7
R_EARTH_KM = 6371.0

# --- corpus inputs ----------------------------------------------------------
H_drift = 2.3e-18          # s^-1, corpus drift rate H_drift ~ H_0 (Paper 0)
T_yr = 12.0                # VLBI scale drift baseline, ~2013-2025
observed_ppb = 0.2         # ITRF2020 VLBI-vs-SLR scale drift (~1.2 mm)

# --- channel 1: VLBI (geometric, matter-metric anchor) ----------------------
# s_VLBI ~ A  ->  full drift participation
s_dot_VLBI = H_drift

# --- channel 2: SLR (dynamic anchor) ----------------------------------------
# Case (a) -- maximal g-anchoring: the GM-anchored orbit carries none of the
# drift and the measured range residuals are absorbed as epoch noise.
s_dot_SLR_a = 0.0
# Case (b) -- Kepler-tracking estimate: the orbit's inferred size follows the
# measured period, a_inf ~ A^(2/3), while ranges ~ A. The station radius is
# the difference r_st = a_inf - rho; for LAGEOS-1/2 (a_g ~ 12,271 km semi-
# major axis, representative slant range rho_g ~ 5,900 km) the logarithmic
# A-response of r_st is (2 a_g/3 - rho_g) / (a_g - rho_g).
a_lageos_km = 6371.0 + 5900.0      # ~5900 km altitude -> 12,271 km
rho_g_km = 5900.0                  # representative zenith-ish slant range
r_st_km = a_lageos_km - rho_g_km   # = R_EARTH for the schematic geometry
dlnr_dlnA = (2.0 * a_lageos_km / 3.0 - rho_g_km) / r_st_km
s_dot_SLR_b = dlnr_dlnA * H_drift  # anchor tracks a partial share of the drift

# --- predicted splits -------------------------------------------------------
def split_ppb(s_v, s_s, years):
    return (s_v - s_s) * years * S_PER_YR * 1e9

pred_max_ppb = split_ppb(s_dot_VLBI, s_dot_SLR_a, T_yr)      # = H_drift * T
pred_kepler_ppb = split_ppb(s_dot_VLBI, s_dot_SLR_b, T_yr)

# --- realized participation fraction ---------------------------------------
f_vs_max = observed_ppb / pred_max_ppb
f_vs_kepler = observed_ppb / pred_kepler_ppb

# --- scale-equivalent lengths ----------------------------------------------
def ppb_to_mm(x):
    return x * 1e-9 * R_EARTH_KM * 1e6  # ppb of Earth radius in mm

ledger = {
    "step": "5.2",
    "status": "success",
    "description": "VLBI-SLR scale-drift channel: anchoring-split derivation, "
                   "predicted amplitude, and the realized drift-participation "
                   "bound from the ITRF2020 observation.",
    "inputs": {
        "H_drift_s^-1": H_drift,
        "baseline_yr": T_yr,
        "observed_drift_ppb": observed_ppb,
        "observed_drift_mm": ppb_to_mm(observed_ppb),
    },
    "anchoring_split": {
        "vlbi": {
            "anchor": "geometric: quasar-fixed positions + matter-clock group "
                      "delays converted by defined c; length unit = "
                      "matter-metric meter",
            "scale_evolution": "s ~ A(t); s_dot/s = A_dot/A = H_drift",
            "s_dot_s^-1": s_dot_VLBI,
        },
        "slr_case_a_g_anchor": {
            "anchor": "dynamic: orbit fixed by adopted GM (drift-free "
                      "conventional constant); ranges absorbed into the "
                      "dynamical solution",
            "s_dot_s^-1": s_dot_SLR_a,
        },
        "slr_case_b_kepler_tracking": {
            "anchor": "dynamic with measured-period tracking: a_inf = "
                      "(GM T_tilde^2/4 pi^2)^(1/3) ~ A^(2/3); ranges ~ A; "
                      "station radius = orbit - range",
            "lageos_semi_major_km": a_lageos_km,
            "slant_range_km": rho_g_km,
            "dlnr_over_dlnA": dlnr_dlnA,
            "s_dot_s^-1": s_dot_SLR_b,
        },
    },
    "predicted_split": {
        "max_g_anchored_ppb_per_12yr": pred_max_ppb,
        "kepler_tracked_ppb_per_12yr": pred_kepler_ppb,
        "max_g_anchored_mm": ppb_to_mm(pred_max_ppb),
        "kepler_tracked_mm": ppb_to_mm(pred_kepler_ppb),
    },
    "participation_bound": {
        "realized_fraction_vs_max": f_vs_max,
        "realized_fraction_vs_kepler": f_vs_kepler,
        "interpretation": "the observed ~0.2 ppb sits below every limiting "
                          "case of the predicted split, so the ITRF2020 drift "
                          "bounds the locally realized share of the ambient "
                          "conformal drift at f ~ 0.2-0.35 rather than "
                          "confirming the channel; full g-anchoring would "
                          "predict ~0.9 ppb/12 yr, Keplerian period tracking "
                          "~0.6 ppb/12 yr. The sign of the observed split "
                          "(VLBI scale drifting high relative to SLR) is the "
                          "direction predicted when the geometrically "
                          "anchored frame carries more of the ambient drift.",
    },
    "rule8_note": "a uniform rescaling is invisible to same-sector "
                  "comparisons; the channel is observable only because the "
                  "two frames anchor scale in different metric sectors, and "
                  "the realized fraction is < 1 precisely because the "
                  "dynamical anchor's own time tags share the drifting "
                  "clocks (internal cancellation).",
}

out = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "outputs",
    "step_5_2_vlbi_slr_scale_drift.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(ledger, f, indent=2)
print(json.dumps(ledger["predicted_split"], indent=2))
print(json.dumps(ledger["participation_bound"], indent=2))
print("wrote", os.path.normpath(out))
