!pip install git+https://github.com/google-research/timesfm.git#egg=timesfm[torch] -q

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import chi2
from scipy.signal import place_poles
import timesfm

# ============================================================
# SYSTEM DEFINITIONS  (single sensor)
# ============================================================
np.random.seed(42)

omega   = 0.3
dt      = 1.0
M       = 1
sigma_n = 0.01
x0      = np.array([1.0, 0.0])

A = np.array([
    [ np.cos(omega*dt),  np.sin(omega*dt)/omega],
    [-np.sin(omega*dt)*omega, np.cos(omega*dt)]
])
C = np.array([[1.0, 0.0]])   # single sensor, shape (1, 2)

# ============================================================
# LOAD TIMESFM
# ============================================================
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(
    timesfm.ForecastConfig(max_context=512, max_horizon=1, normalize_inputs=True)
)
print("Model loaded.")

# ============================================================
# PARAMETERS
# ============================================================
N_trials        = 10           # set to 20 for final run
context_len     = 50
alpha_det       = 0.005
buffer_weight   = 0.8

T_warmup        = context_len + 1
T_clean         = 30
attack_duration = 20
T_att           = attack_duration
attack_start    = T_warmup + T_clean
T_trial         = T_warmup + T_clean + attack_duration
T_online_trial  = T_clean + attack_duration
threshold       = chi2.ppf(1 - alpha_det, df=M)
eta             = 0.09 * threshold
q               = np.array([1.0, 0.0])

assert context_len <= T_warmup
print(f"Threshold: {threshold:.4f} | eta: {eta:.4f}")

# ============================================================
# LUENBERGER OBSERVER
# ============================================================
desired_poles = np.array([0.5 + 1e-1j, 0.5 - 1e-1j])
result        = place_poles(A.T, C.T, desired_poles)
K_obs         = result.gain_matrix.T   # shape (2, 1)
print(f"Observer gain K_obs:\n{K_obs}")

# ============================================================
# PRECOMPUTE c_k
# ============================================================
c_k_list = []
for k in range(T_att):
    Apow = np.linalg.matrix_power(A, T_att - 1 - k)
    c_k  = K_obs.T @ Apow.T @ q   # shape (1,)
    c_k_list.append(c_k)

# ============================================================
# CLEAN RUN — estimate Sigma_fm and Sigma_obs
# ============================================================
np.random.seed(0)
T_init = T_warmup + T_clean
x_init = np.zeros((T_init + 1, 2));  x_init[0] = x0
y_init = np.zeros((T_init, M))

for t in range(T_init):
    x_init[t+1] = A @ x_init[t]
    y_init[t]   = C @ x_init[t] + sigma_n * np.random.randn(M)

# TimesFM residuals on clean run — batched
contexts_init = [y_init[max(0, T_warmup + t - context_len): T_warmup + t, 0]
                 for t in range(T_clean)]
pf_init, _ = model.forecast(horizon=1, inputs=contexts_init)
res_fm_init = y_init[T_warmup:T_warmup + T_clean, 0] - pf_init[:, 0]

Sigma_fm     = np.atleast_2d(np.var(res_fm_init))
Sigma_fm_inv = 1.0 / Sigma_fm

# Observer residuals on clean run
x_est_s = np.zeros(2)
for t in range(T_warmup):
    innov   = y_init[t, 0] - (C @ x_est_s)[0]
    x_est_s = A @ x_est_s + K_obs[:, 0] * innov

res_obs_init = np.zeros(T_clean)
for t in range(T_clean):
    global_t        = T_warmup + t
    innov           = y_init[global_t, 0] - (C @ x_est_s)[0]
    x_est_s         = A @ x_est_s + K_obs[:, 0] * innov
    res_obs_init[t] = innov

Sigma_obs     = np.atleast_2d(np.var(res_obs_init))
Sigma_obs_inv = 1.0 / Sigma_obs

print(f"Sigma_fm: {Sigma_fm[0,0]:.6f} | Sigma_obs: {Sigma_obs[0,0]:.6f}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def simulate_epsilon(delta_seq):
    """Analytically compute w^T epsilon[T] for a given delta sequence."""
    eps = np.zeros(2)
    for k in range(T_att):
        eps = (A - K_obs @ C) @ eps + K_obs[:, 0] * delta_seq[k]
    return q @ eps

def build_attacked_trajectory(y_cln_full, delta_seq):
    """Inject delta_seq into clean trajectory, return attacked 1-D signal."""
    y_att_s = y_cln_full[:, 0].copy()
    eps     = np.zeros(2)
    for k in range(T_att):
        t          = attack_start + k
        a_t        = delta_seq[k] + (C @ eps)[0]
        y_att_s[t] += a_t
        eps        = (A - K_obs @ C) @ eps + K_obs[:, 0] * delta_seq[k]
    return y_att_s

def query_gs_far(y_cln_full, delta_seq):
    """
    Inject delta_seq, run TimesFM with buffer protection (batched per step),
    return empirical FAR over attack window.
    Attacker is stealthy if returned FAR <= alpha_det.
    """
    y_att_s = build_attacked_trajectory(y_cln_full, delta_seq)

    # batched TimesFM call over all attack steps
    contexts = [y_att_s[max(0, attack_start + k - context_len): attack_start + k]
                for k in range(T_att)]
    pf, _ = model.forecast(horizon=1, inputs=contexts)
    y_hats = pf[:, 0]   # shape (T_att,)

    residuals = y_att_s[attack_start: attack_start + T_att] - y_hats
    g_s_vals  = residuals**2 * Sigma_fm_inv[0, 0]
    n_alarms  = int(np.sum(g_s_vals > threshold))

    return n_alarms / T_att

def project_gp(delta_seq):
    """Project each delta_k onto g_p feasible set (scalar clipping)."""
    delta_proj = delta_seq.copy()
    for k in range(T_att):
        scale = delta_seq[k]**2 * Sigma_obs_inv[0, 0]
        if scale > eta:
            delta_proj[k] = np.sign(delta_seq[k]) * np.sqrt(
                eta / Sigma_obs_inv[0, 0]
            )
    return delta_proj

def plot_band(ax, chi2_trials, color, title, t_offset=0):
    t_axis = np.arange(chi2_trials.shape[1]) + t_offset
    mean   = chi2_trials.mean(axis=0)
    std    = chi2_trials.std(axis=0)
    for trial in range(chi2_trials.shape[0]):
        ax.plot(t_axis, chi2_trials[trial], color=color, lw=0.5, alpha=0.3)
    ax.plot(t_axis, mean, '-', color=color, lw=2.0, label='Mean')
    ax.fill_between(t_axis, mean - std, mean + std,
                    alpha=0.2, color=color, label='±1 std')
    ax.axhline(threshold, color='red', linestyle='--', lw=1.5,
               label=f'Threshold ($\\alpha$={alpha_det})')
    ax.set_ylabel('$\\chi^2$ statistic')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# ============================================================
# BLOCK 1: OBLIVIOUS ATTACK (Theorem 1) — Monte Carlo
# ============================================================
chi2_fm_att_trials  = np.zeros((N_trials, attack_duration))
chi2_obs_att_trials = np.zeros((N_trials, attack_duration))
chi2_fm_cln_trials  = np.zeros((N_trials, attack_duration))
chi2_obs_cln_trials = np.zeros((N_trials, attack_duration))
state_dev_oblivious = np.zeros(N_trials)
y_plot_att = None
y_plot_cln = None

# delta_oblivious is the same across trials (deterministic, depends only on Sigma_obs)
delta_oblivious = np.zeros(T_att)
for k in range(T_att):
    c_k   = c_k_list[k][0]
    sig   = Sigma_obs[0, 0]
    denom = np.sqrt(c_k**2 * sig)
    if denom > 1e-10:
        delta_oblivious[k] = np.sqrt(eta) * np.sqrt(sig) * c_k / denom

print(f"\nBlock 1: Oblivious attack | N_trials={N_trials} | eta={eta:.3f} ...")

for trial in range(N_trials):
    np.random.seed(trial + 1)

    x_trial = np.zeros((T_trial + 1, 2));  x_trial[0] = x0
    y_cln   = np.zeros((T_trial, M))
    for t in range(T_trial):
        x_trial[t+1] = A @ x_trial[t]
        y_cln[t]     = C @ x_trial[t] + sigma_n * np.random.randn(M)

    state_dev_oblivious[trial] = simulate_epsilon(delta_oblivious)

    # inject attack into trajectory
    epsilon = np.zeros(2)
    y_att   = y_cln.copy()
    for t in range(T_trial):
        if attack_start <= t < attack_start + attack_duration:
            k_att   = t - attack_start
            c_k     = c_k_list[k_att]
            Sigma_c = Sigma_obs @ c_k
            denom   = np.sqrt(c_k @ Sigma_obs @ c_k)
            delta_k = np.sqrt(eta) * Sigma_c / denom if denom > 1e-10 else np.zeros(M)
            a_t     = delta_k + C @ epsilon
            y_att[t] = y_cln[t] + a_t
            epsilon  = (A - K_obs @ C) @ epsilon + K_obs @ a_t
        else:
            epsilon = (A - K_obs @ C) @ epsilon

    # TimesFM residuals with buffer protection — sequential (needs buffer update)
    res_fm_att = np.zeros((T_online_trial, M))
    res_fm_cln = np.zeros((T_online_trial, M))
    y_buffer   = y_att.copy()

    for t in range(T_warmup, T_trial):
        online_t = t - T_warmup
        start    = max(0, t - context_len)

        pf_att, _               = model.forecast(horizon=1, inputs=[y_buffer[start:t, 0]])
        y_hat                   = pf_att[0, 0]
        res_fm_att[online_t, 0] = y_att[t, 0] - y_hat
        chi2_now                = res_fm_att[online_t, 0]**2 * Sigma_fm_inv[0, 0]
        if chi2_now > threshold and t >= attack_start:
            y_buffer[t, 0] = buffer_weight * y_hat + (1 - buffer_weight) * y_att[t, 0]
        else:
            y_buffer[t, 0] = y_att[t, 0]

        pf_cln, _               = model.forecast(horizon=1, inputs=[y_cln[start:t, 0]])
        res_fm_cln[online_t, 0] = y_cln[t, 0] - pf_cln[0, 0]

    # observer residuals
    res_obs_att = np.zeros((T_online_trial, M))
    res_obs_cln = np.zeros((T_online_trial, M))
    x_est_att   = np.zeros(2);  x_est_cln = np.zeros(2)

    for t in range(T_trial):
        innov_att            = y_att[t, 0] - (C @ x_est_att)[0]
        x_est_att            = A @ x_est_att + K_obs[:, 0] * innov_att
        innov_cln            = y_cln[t, 0] - (C @ x_est_cln)[0]
        x_est_cln            = A @ x_est_cln + K_obs[:, 0] * innov_cln
        if t >= T_warmup:
            online_t                 = t - T_warmup
            res_obs_att[online_t, 0] = innov_att
            res_obs_cln[online_t, 0] = innov_cln

    chi2_fm_att_trials[trial]  = res_fm_att[T_clean:, 0]**2  * Sigma_fm_inv[0, 0]
    chi2_fm_cln_trials[trial]  = res_fm_cln[T_clean:, 0]**2  * Sigma_fm_inv[0, 0]
    chi2_obs_att_trials[trial] = res_obs_att[T_clean:, 0]**2 * Sigma_obs_inv[0, 0]
    chi2_obs_cln_trials[trial] = res_obs_cln[T_clean:, 0]**2 * Sigma_obs_inv[0, 0]

    if trial == 0:
        y_plot_att = y_att.copy()
        y_plot_cln = y_cln.copy()

    print(f"  Trial {trial+1}/{N_trials} done")

FAR_fm  = np.mean(chi2_fm_cln_trials  > threshold)
FAR_obs = np.mean(chi2_obs_cln_trials > threshold)
DR_fm   = np.mean(np.any(chi2_fm_att_trials  > threshold, axis=1))
DR_obs  = np.mean(np.any(chi2_obs_att_trials > threshold, axis=1))
print(f"\nBlock 1 done.")
print(f"TimesFM  — FAR: {FAR_fm:.3f} | DR: {DR_fm:.3f}")
print(f"Observer — FAR: {FAR_obs:.3f} | DR: {DR_obs:.3f}")

# ============================================================
# PLOT 1: OBLIVIOUS ATTACK — immediately after Block 1
# ============================================================
fig1 = plt.figure(figsize=(14, 9))
gs1  = gridspec.GridSpec(3, 2, figure=fig1)

# row 0: sensor trajectory (full width)
ax0 = fig1.add_subplot(gs1[0, :])
t_full = np.arange(len(y_plot_att))
ax0.plot(t_full, y_plot_att[:, 0], '-', color='steelblue',
         lw=0.8, label='Sensor (attacked)')
ax0.plot(t_full, y_plot_cln[:, 0], '--', color='gray',
         lw=0.8, alpha=0.6, label='Sensor (clean)')
ax0.axvline(attack_start, color='red', linestyle='--', lw=1.5, label='Attack start')
ax0.axvline(attack_start + attack_duration, color='darkred',
            linestyle='--', lw=1.5, label='Attack end')
ax0.axvspan(attack_start, attack_start + attack_duration, alpha=0.08, color='red')
ax0.set_xlabel('Time step')
ax0.set_ylabel('Sensor output')
ax0.set_title('Sensor trajectory — oblivious attack vs clean')
ax0.legend(fontsize=8);  ax0.grid(alpha=0.3)

# row 1: TimesFM
ax_fm_cln = fig1.add_subplot(gs1[1, 0])
ax_fm_att = fig1.add_subplot(gs1[1, 1])
plot_band(ax_fm_cln, chi2_fm_cln_trials, 'steelblue',
          'TimesFM — no attack', t_offset=attack_start)
plot_band(ax_fm_att, chi2_fm_att_trials, 'steelblue',
          f'TimesFM — oblivious attack | DR={DR_fm:.2f}', t_offset=attack_start)
ax_fm_cln.set_xlabel('Time step')
ax_fm_att.set_xlabel('Time step in attack window')

# row 2: Observer
ax_obs_cln = fig1.add_subplot(gs1[2, 0])
ax_obs_att = fig1.add_subplot(gs1[2, 1])
plot_band(ax_obs_cln, chi2_obs_cln_trials, 'darkorange',
          'Observer — no attack', t_offset=attack_start)
plot_band(ax_obs_att, chi2_obs_att_trials, 'darkorange',
          f'Observer — oblivious attack | DR={DR_obs:.2f}', t_offset=attack_start)
ax_obs_cln.set_xlabel('Time step')
ax_obs_att.set_xlabel('Time step in attack window')

# equalise y-axes
y_max_cln = max(ax_fm_cln.get_ylim()[1], ax_obs_cln.get_ylim()[1])
for ax in [ax_fm_cln, ax_obs_cln]:
    ax.set_ylim(0, y_max_cln)
y_max_att = max(ax_fm_att.get_ylim()[1], ax_obs_att.get_ylim()[1])
for ax in [ax_fm_att, ax_obs_att]:
    ax.set_ylim(0, y_max_att)

fig1.suptitle(
    f'Oblivious attack | N={N_trials} | ctx={context_len} | '
    f'FAR $\\alpha$={alpha_det} | $\\eta$={eta:.3f}',
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig('oblivious_attack.pdf', bbox_inches='tight')
plt.show()
print("Plot 1 saved.")

# ============================================================
# BLOCK 2: ADAPTIVE ATTACK (SPSA + FAR-based binary search)
# ============================================================
N_bisect  = 8    # binary search steps per SPSA iteration
N_spsa    = 20   # SPSA iterations
eta_spsa  = 0.05
eps_spsa  = 0.005

state_dev_adaptive = np.zeros(N_trials)
state_dev_noattack = np.zeros(N_trials)   # always zero by definition
beta_init_log      = np.zeros(N_trials)   # log scaling factor at warm start

print(f"\nBlock 2: Adaptive SPSA | N_trials={N_trials} | "
      f"N_spsa={N_spsa} | N_bisect={N_bisect} ...")

for trial in range(N_trials):
    np.random.seed(trial + 1)

    x_trial     = np.zeros((T_trial + 1, 2));  x_trial[0] = x0
    y_cln_trial = np.zeros((T_trial, M))
    for t in range(T_trial):
        x_trial[t+1]   = A @ x_trial[t]
        y_cln_trial[t] = C @ x_trial[t] + sigma_n * np.random.randn(M)

    # warm start from oblivious attack
    delta = delta_oblivious.copy()

    # ensure warm start is feasible w.r.t. g_s
    # beta=0 -> zero attack (FAR=0, always feasible)
    # beta=1 -> delta_oblivious (may be infeasible)
    far_init = query_gs_far(y_cln_trial, delta)
    if far_init > alpha_det:
        beta_lo, beta_hi = 0.0, 1.0
        for _ in range(N_bisect):
            beta_mid  = (beta_lo + beta_hi) / 2.0
            delta_mid = beta_mid * delta
            far       = query_gs_far(y_cln_trial, delta_mid)
            if far <= alpha_det:
                beta_lo = beta_mid
            else:
                beta_hi = beta_mid
        delta = beta_lo * delta
        beta_init_log[trial] = beta_lo
        print(f"  Trial {trial+1}: warm start FAR={far_init:.3f} "
              f"— scaled to beta={beta_lo:.3f}")
    else:
        beta_init_log[trial] = 1.0
        print(f"  Trial {trial+1}: warm start feasible (FAR={far_init:.3f})")

    # SPSA loop
    for spsa_iter in range(N_spsa):
        perturbation = np.random.choice([-1.0, 1.0], size=T_att)
        d_plus       = project_gp(delta + eps_spsa * perturbation)
        d_minus      = project_gp(delta - eps_spsa * perturbation)
        J_plus       = simulate_epsilon(d_plus)
        J_minus      = simulate_epsilon(d_minus)
        grad         = (J_plus - J_minus) / (2 * eps_spsa * perturbation)

        delta_new = project_gp(delta + eta_spsa * grad)

        # binary search: largest beta s.t. FAR <= alpha_det
        # beta=0 -> current delta (feasible by invariant)
        # beta=1 -> delta_new (may be infeasible)
        beta_lo, beta_hi = 0.0, 1.0
        for _ in range(N_bisect):
            beta_mid  = (beta_lo + beta_hi) / 2.0
            delta_mid = (1 - beta_mid) * delta + beta_mid * delta_new
            far       = query_gs_far(y_cln_trial, delta_mid)
            if far <= alpha_det:
                beta_lo = beta_mid
            else:
                beta_hi = beta_mid

        delta = (1 - beta_lo) * delta + beta_lo * delta_new

    state_dev_adaptive[trial] = simulate_epsilon(delta)
    print(f"  Trial {trial+1}/{N_trials} — "
          f"oblivious: {state_dev_oblivious[trial]:.4f} | "
          f"adaptive:  {state_dev_adaptive[trial]:.4f}")

print("\nBlock 2 done.")

# ============================================================
# PLOT 2: BOX PLOT — oblivious vs adaptive state deviation
# ============================================================
fig2, ax2 = plt.subplots(figsize=(7, 5))

data_box = [state_dev_noattack, state_dev_oblivious, state_dev_adaptive]
labels   = ['No attack', 'Oblivious\n(Theorem 1)', 'Adaptive\n(SPSA)']
colors   = ['steelblue', 'darkorange', 'seagreen']

bp = ax2.boxplot(data_box, patch_artist=True, widths=0.5,
                 medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(labels, fontsize=11)
ax2.set_ylabel(r'State deviation $w^\top \epsilon_p[T]$', fontsize=11)
ax2.set_title(
    f'Simultaneous evasion | Single sensor | '
    f'$N$={N_trials} | SPSA iters={N_spsa} | FAR $\\alpha$={alpha_det}',
    fontsize=10
)
ax2.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('adaptive_attack_boxplot.pdf', bbox_inches='tight')
plt.show()
print("Plot 2 saved.")