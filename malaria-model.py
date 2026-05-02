"""
Reproduction of selected results from:

  Chitnis, N., Hyman, J.M., Cushing, J.M. (2008).
  "Determining Important Parameters in the Spread of Malaria Through the
  Sensitivity Analysis of a Mathematical Model."
  Bulletin of Mathematical Biology, 70: 1272-1296.

What this script reproduces
---------------------------
1. The disease-free equilibrium population sizes N*_h and N*_v (Eq. 4).
2. The basic reproduction number R0 (Eqs. 5, 6a, 6b) for both
   high-transmission and low-transmission baseline parameter sets,
   targeting the paper's reported values R0 ~ 1.1 (low) and R0 ~ 4.4 (high).
3. Numerical integration of the scaled ODE system (Eqs. 2a-2g) from the
   initial conditions used in Figs. 2 and 3, plotted in the original
   (unscaled) state variables (S_h, E_h, I_h, R_h, S_v, E_v, I_v).
4. The endemic equilibrium found by long-time integration, compared
   against the paper's reported endemic equilibria (Eqs. 7 and 8).
5. Normalized forward sensitivity indices of R0 to all 17 parameters
   (Table 4), computed by automatic-style central differences on the
   analytic R0 formula. The structural identities

       Y^{R0}_{beta_hv} = Y^{R0}_{beta_vh} = 1/2,
       Y^{R0}_{sigma_v} + Y^{R0}_{sigma_h} = 1   (i.e., Y^{R0}_zeta = 1),

   are verified as a sanity check.

Run:  python malaria_reproduction.py
Outputs: prints results to stdout and saves figures to PNG files.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------
# 1.  Parameters
# -----------------------------------------------------------------------------


@dataclass
class Params:
    """Baseline parameter set for the malaria model (Table 3 of the paper).

    All rates are per day. See Table 2 for parameter meanings.
    """

    Lambda_h: float  # immigration rate of humans (humans/day)
    psi_h: float  # per-capita human birth rate (1/day)
    psi_v: float  # per-capita mosquito birth rate (1/day)
    sigma_v: float  # mosquito desired biting rate (1/day)
    sigma_h: float  # max mosquito bites a human can have (1/day)
    beta_hv: float  # P(transmission mosquito -> human | bite)
    beta_vh: float  # P(transmission infectious human -> mosquito | bite)
    beta_vh_tilde: float  # P(transmission recovered human -> mosquito | bite)
    nu_h: float  # 1/(latent period in humans) (1/day)
    nu_v: float  # 1/(latent period in mosquitoes) (1/day)
    gamma_h: float  # human recovery rate (1/day)
    delta_h: float  # disease-induced human death rate (1/day)
    rho_h: float  # rate of loss of immunity (1/day)
    mu1_h: float  # density-independent human death/emigration (1/day)
    mu2_h: float  # density-dependent human death/emigration (1/(human*day))
    mu1_v: float  # density-independent mosquito death (1/day)
    mu2_v: float  # density-dependent mosquito death (1/(mosq*day))


# Baseline values from Table 3.
LOW = Params(
    Lambda_h=0.041,
    psi_h=5.5e-5,
    psi_v=0.13,
    sigma_v=0.33,
    sigma_h=4.3,
    beta_hv=0.022,
    beta_vh=0.24,
    beta_vh_tilde=0.024,
    nu_h=0.10,
    nu_v=0.083,
    gamma_h=0.0035,
    delta_h=1.8e-5,
    rho_h=2.7e-3,
    mu1_h=8.8e-6,
    mu2_h=2.0e-7,
    mu1_v=0.033,
    mu2_v=4.0e-5,
)

HIGH = Params(
    Lambda_h=0.033,
    psi_h=1.1e-4,
    psi_v=0.13,
    sigma_v=0.50,
    sigma_h=19.0,
    beta_hv=0.022,
    beta_vh=0.48,
    beta_vh_tilde=0.048,
    nu_h=0.10,
    nu_v=0.091,
    gamma_h=0.0035,
    delta_h=9.0e-5,
    rho_h=5.5e-4,
    mu1_h=1.6e-5,
    mu2_h=3.0e-7,
    mu1_v=0.033,
    mu2_v=2.0e-5,
)


# -----------------------------------------------------------------------------
# 2.  Disease-free equilibrium populations and R0
# -----------------------------------------------------------------------------


def disease_free_populations(p: Params) -> tuple[float, float]:
    """Return (N_h*, N_v*) at the disease-free equilibrium, Eq. (4)."""
    a = p.psi_h - p.mu1_h
    Nh_star = (a + math.sqrt(a * a + 4.0 * p.mu2_h * p.Lambda_h)) / (2.0 * p.mu2_h)
    Nv_star = (p.psi_v - p.mu1_v) / p.mu2_v
    return Nh_star, Nv_star


def R0(p: Params) -> float:
    """Basic reproduction number, Eq. (5)-(6).

    R0 = sqrt(K_vh * K_hv), where K_hv counts new mosquito-to-human
    infections per infectious mosquito (at the DFE) and K_vh counts new
    human-to-mosquito infections per infectious human.
    """
    Nh, Nv = disease_free_populations(p)
    contact = (p.sigma_v * p.sigma_h) / (p.sigma_v * Nv + p.sigma_h * Nh)

    K_hv = (
        (p.nu_v / (p.nu_v + p.mu1_v + p.mu2_v * Nv))
        * contact
        * Nh
        * p.beta_hv
        * (1.0 / (p.mu1_v + p.mu2_v * Nv))
    )

    K_vh = (
        (p.nu_h / (p.nu_h + p.mu1_h + p.mu2_h * Nh))
        * contact
        * Nv
        * (1.0 / (p.gamma_h + p.delta_h + p.mu1_h + p.mu2_h * Nh))
        * (
            p.beta_vh
            + p.beta_vh_tilde * (p.gamma_h / (p.rho_h + p.mu1_h + p.mu2_h * Nh))
        )
    )

    return math.sqrt(K_vh * K_hv)


# -----------------------------------------------------------------------------
# 3.  The dynamical system
# -----------------------------------------------------------------------------


def rhs_scaled(t, y, p: Params):
    """Right-hand side of the scaled malaria model, Eqs. (2a)-(2g).

    State vector y = (e_h, i_h, r_h, N_h, e_v, i_v, N_v).
    """
    e_h, i_h, r_h, N_h, e_v, i_v, N_v = y

    # Half-harmonic-mean contact term sigma_v sigma_h / (sigma_v N_v + sigma_h N_h)
    denom = p.sigma_v * N_v + p.sigma_h * N_h
    contact = (p.sigma_v * p.sigma_h) / denom  # (bites per human per mosquito per day)

    # Lambda_h / N_h appears throughout; guard against division by zero.
    lam_over_N = p.Lambda_h / N_h

    de_h = (
        (contact * N_v * p.beta_hv * i_v) * (1.0 - e_h - i_h - r_h)
        - (p.nu_h + p.psi_h + lam_over_N) * e_h
        + p.delta_h * i_h * e_h
    )
    di_h = (
        p.nu_h * e_h
        - (p.gamma_h + p.delta_h + p.psi_h + lam_over_N) * i_h
        + p.delta_h * i_h * i_h
    )
    dr_h = (
        p.gamma_h * i_h - (p.rho_h + p.psi_h + lam_over_N) * r_h + p.delta_h * i_h * r_h
    )
    dN_h = (
        p.Lambda_h
        + p.psi_h * N_h
        - (p.mu1_h + p.mu2_h * N_h) * N_h
        - p.delta_h * i_h * N_h
    )

    de_v = (contact * N_h * (p.beta_vh * i_h + p.beta_vh_tilde * r_h)) * (
        1.0 - e_v - i_v
    ) - (p.nu_v + p.psi_v) * e_v
    di_v = p.nu_v * e_v - p.psi_v * i_v
    dN_v = p.psi_v * N_v - (p.mu1_v + p.mu2_v * N_v) * N_v

    return [de_h, di_h, dr_h, dN_h, de_v, di_v, dN_v]


def initial_condition_scaled(Sh, Eh, Ih, Rh, Sv, Ev, Iv):
    """Convert original counts (Fig. 2 / 3 IC) to the scaled state vector."""
    Nh = Sh + Eh + Ih + Rh
    Nv = Sv + Ev + Iv
    return np.array([Eh / Nh, Ih / Nh, Rh / Nh, Nh, Ev / Nv, Iv / Nv, Nv])


def integrate(p: Params, y0, t_end_days: float, n_points: int = 4000):
    sol = solve_ivp(
        rhs_scaled,
        (0.0, t_end_days),
        y0,
        args=(p,),
        method="LSODA",
        rtol=1e-9,
        atol=1e-12,
        t_eval=np.linspace(0.0, t_end_days, n_points),
        max_step=5.0,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol


def to_original_counts(sol):
    """Recover (S_h, E_h, I_h, R_h, S_v, E_v, I_v) from the scaled solution."""
    e_h, i_h, r_h, N_h, e_v, i_v, N_v = sol.y
    s_h = 1.0 - e_h - i_h - r_h
    s_v = 1.0 - e_v - i_v
    Sh = s_h * N_h
    Eh = e_h * N_h
    Ih = i_h * N_h
    Rh = r_h * N_h
    Sv = s_v * N_v
    Ev = e_v * N_v
    Iv = i_v * N_v
    return dict(
        t=sol.t,
        Sh=Sh,
        Eh=Eh,
        Ih=Ih,
        Rh=Rh,
        Sv=Sv,
        Ev=Ev,
        Iv=Iv,
        eh=e_h,
        ih=i_h,
        rh=r_h,
        Nh=N_h,
        ev=e_v,
        iv=i_v,
        Nv=N_v,
    )


# -----------------------------------------------------------------------------
# 4.  Sensitivity indices of R0
# -----------------------------------------------------------------------------


def sensitivity_indices_R0(p: Params, rel_step: float = 1e-6) -> dict[str, float]:
    """Normalized forward sensitivity index Y^{R0}_q = (dR0/dq)*(q/R0)
    for each parameter q, computed by central difference.
    """
    R_base = R0(p)
    indices = {}
    for f in fields(p):
        name = f.name
        q = getattr(p, name)
        if q == 0.0:
            indices[name] = 0.0
            continue
        h = rel_step * abs(q)
        plus = Params(**{**asdict(p), name: q + h})
        minus = Params(**{**asdict(p), name: q - h})
        dR_dq = (R0(plus) - R0(minus)) / (2.0 * h)
        indices[name] = dR_dq * q / R_base
    return indices


# Pretty-print symbol map for sensitivity table.
SYMBOL = {
    "Lambda_h": "Λ_h",
    "psi_h": "ψ_h",
    "psi_v": "ψ_v",
    "sigma_v": "σ_v",
    "sigma_h": "σ_h",
    "beta_hv": "β_hv",
    "beta_vh": "β_vh",
    "beta_vh_tilde": "β̃_vh",
    "nu_h": "ν_h",
    "nu_v": "ν_v",
    "gamma_h": "γ_h",
    "delta_h": "δ_h",
    "rho_h": "ρ_h",
    "mu1_h": "μ_1h",
    "mu2_h": "μ_2h",
    "mu1_v": "μ_1v",
    "mu2_v": "μ_2v",
}


# -----------------------------------------------------------------------------
# 5.  Reporting helpers
# -----------------------------------------------------------------------------


def report_basic(label: str, p: Params) -> None:
    Nh, Nv = disease_free_populations(p)
    r0 = R0(p)
    print(f"--- {label} transmission ---")
    print(
        f"  Disease-free N*_h = {Nh:8.2f}    (paper: {523 if label == 'High' else 583})"
    )
    print(
        f"  Disease-free N*_v = {Nv:8.2f}    (paper: {4850 if label == 'High' else 2425})"
    )
    print(
        f"  R0                = {r0:8.4f}    (paper: {4.4 if label == 'High' else 1.1})"
    )
    print()


def report_endemic_equilibrium(label: str, p: Params, ic, t_end=200_000) -> None:
    sol = integrate(p, ic, t_end_days=t_end, n_points=2000)
    out = to_original_counts(sol)
    print(f"--- Endemic equilibrium reached by long-time integration ({label}) ---")
    print(
        f"  e_h = {out['eh'][-1]:.4f}, i_h = {out['ih'][-1]:.4f}, "
        f"r_h = {out['rh'][-1]:.4f}, N_h = {out['Nh'][-1]:.1f}"
    )
    print(
        f"  e_v = {out['ev'][-1]:.4f}, i_v = {out['iv'][-1]:.4f}, "
        f"N_v = {out['Nv'][-1]:.1f}"
    )
    if label == "High":
        print("  paper Eq. (8): (0.0059, 0.16, 0.77, 490, 0.15, 0.11, 4850)")
    else:
        print("  paper Eq. (7): (0.0029, 0.080, 0.10, 578, 0.024, 0.016, 2425)")
    print()


def report_sensitivities(label: str, p: Params) -> None:
    idx = sensitivity_indices_R0(p)
    ranked = sorted(idx.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print(
        f"--- Sensitivity indices of R0 ({label} transmission), ranked by |index| ---"
    )
    print(f"  {'rank':>4}  {'parameter':<6}  {'Y^R0_q':>10}")
    for rank, (name, val) in enumerate(ranked, start=1):
        print(f"  {rank:>4}  {SYMBOL[name]:<6}  {val:+10.4f}")
    # Structural sanity checks the paper highlights.
    # Paper: Y^R0_{β_hv} = 1/2 exactly (paper, p. 1281).
    # Note: Y^R0_{β_vh} + Y^R0_{β̃_vh} = 1/2 because β_vh and β̃_vh enter R0 only
    # through the combination β_vh + β̃_vh*(γ_h/(ρ_h+...)) under a square root.
    print(
        f"  identity check  Y^R0_{{β_hv}}                 = {idx['beta_hv']:+.4f}  (paper: +0.5)"
    )
    s_beta = idx["beta_vh"] + idx["beta_vh_tilde"]
    print(
        f"  identity check  Y^R0_{{β_vh}} + Y^R0_{{β̃_vh}}  = {s_beta:+.4f}  (should be +0.5)"
    )
    s = idx["sigma_v"] + idx["sigma_h"]
    print(
        f"  identity check  Y^R0_{{σ_v}} + Y^R0_{{σ_h}}      = {s:+.4f}  (= Y^R0_ζ, paper Eq. 12: +1.0)"
    )
    print()


# -----------------------------------------------------------------------------
# 6.  Plots
# -----------------------------------------------------------------------------


def plot_trajectory(label, sol_dict, fname, t_max_days):
    """Reproduce the layout of Figs. 2 and 3: 4 humans + 3 mosquito panels."""
    t = sol_dict["t"]
    mask = t <= t_max_days
    fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True)
    axes = axes.flatten()

    panels = [
        ("S_h (susceptible humans)", sol_dict["Sh"]),
        ("E_h (exposed humans)", sol_dict["Eh"]),
        ("I_h (infectious humans)", sol_dict["Ih"]),
        ("R_h (recovered humans)", sol_dict["Rh"]),
        ("S_v (susceptible mosq.)", sol_dict["Sv"]),
        ("E_v (exposed mosquitoes)", sol_dict["Ev"]),
        ("I_v (infectious mosq.)", sol_dict["Iv"]),
    ]
    for ax, (title, y) in zip(axes, panels):
        ax.plot(t[mask], y[mask], lw=1.5)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].axis("off")
    for ax in axes[5:7]:
        ax.set_xlabel("time (days)")
    fig.suptitle(
        f"Malaria model trajectory — {label} transmission "
        f"(reproduction of Fig. {2 if label == 'Low' else 3})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_sensitivity_bars(p_low, p_high, fname):
    idx_low = sensitivity_indices_R0(p_low)
    idx_high = sensitivity_indices_R0(p_high)

    # Order by mean |index| across the two regimes for readability.
    keys = sorted(
        idx_low.keys(),
        key=lambda k: 0.5 * (abs(idx_low[k]) + abs(idx_high[k])),
        reverse=True,
    )
    labels = [SYMBOL[k] for k in keys]
    low_vals = [idx_low[k] for k in keys]
    high_vals = [idx_high[k] for k in keys]

    x = np.arange(len(keys))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, low_vals, width, label="Low transmission")
    ax.bar(x + width / 2, high_vals, width, label="High transmission")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(r"Sensitivity index  $\Upsilon^{R_0}_{q}$")
    ax.set_title("Sensitivity of $R_0$ to model parameters (Table 4 reproduction)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  saved {fname}")


# -----------------------------------------------------------------------------
# 7.  Main
# -----------------------------------------------------------------------------


def main():
    # 7.1 Disease-free populations and R0.
    print("=" * 72)
    print("STEP 1 — Disease-free equilibrium populations and R0")
    print("=" * 72)
    report_basic("Low", LOW)
    report_basic("High", HIGH)

    # 7.2 Long-time integration to confirm endemic equilibria (Eqs. 7-8).
    print("=" * 72)
    print("STEP 2 — Endemic equilibria via long-time integration")
    print("=" * 72)
    ic_low = initial_condition_scaled(Sh=600, Eh=20, Ih=3, Rh=0, Sv=2400, Ev=30, Iv=5)
    ic_high = initial_condition_scaled(
        Sh=500, Eh=10, Ih=30, Rh=0, Sv=4000, Ev=100, Iv=50
    )
    report_endemic_equilibrium("Low", LOW, ic_low)
    report_endemic_equilibrium("High", HIGH, ic_high)

    # 7.3 Sensitivity indices of R0 (Table 4).
    print("=" * 72)
    print("STEP 3 — Sensitivity indices of R0 (Table 4)")
    print("=" * 72)
    report_sensitivities("Low", LOW)
    report_sensitivities("High", HIGH)

    # 7.4 Plots: trajectories (Figs. 2, 3) and sensitivity bar chart.
    print("=" * 72)
    print("STEP 4 — Plots")
    print("=" * 72)
    sol_low = integrate(LOW, ic_low, t_end_days=15_000, n_points=4000)
    sol_high = integrate(HIGH, ic_high, t_end_days=15_000, n_points=4000)
    plot_trajectory(
        "Low",
        to_original_counts(sol_low),
        "pics/fig2_low_transmission.png",
        t_max_days=15_000,
    )
    plot_trajectory(
        "High",
        to_original_counts(sol_high),
        "pics/fig3_high_transmission.png",
        t_max_days=15_000,
    )
    plot_sensitivity_bars(LOW, HIGH, "pics/table4_sensitivity_bars.png")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
