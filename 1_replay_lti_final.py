!pip install git+https://github.com/google-research/timesfm.git#egg=timesfm[torch] -q

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import chi2
from scipy.signal import place_poles
import timesfm

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
# SYSTEM DEFINITION
# ============================================================
np.random.seed(42)

omega   = 0.3
dt      = 1.0
n_state = 2
M       = 3
sigma_n = 0.01
x0      = np.array([1.0, 0.0])

A = np.array([
    [ np.cos(omega*dt),  np.sin(omega*dt)/omega],
    [-np.sin(omega*dt)*omega, np.cos(omega*dt)]
])

C = np.array([
    [ 1.00,  0.00],
    [ 0.31, -0.48],
    [-0.21,  0.43]
])

# ============================================================
# LUENBERGER OBSERVER — static gain via pole placement
# Observer: xhat[k+1] = A @ xhat[k] + K_obs @ (y[k] - C @ xhat[k])
# Poles placed inside unit circle for stable error dynamics
# ============================================================
desired_poles = np.array([0.5 + 1e-1j, 0.5 - 1e-1j])   # conjugate pair for n=2
result        = place_poles(A.T, C.T, desired_poles)
K_obs         = result.gain_matrix.T   # shape (n, M)

print(f"Observer poles: {desired_poles}")
print(f"K_obs shape: {K_obs.shape}")

# ============================================================
# PARAMETERS
# ============================================================
N_trials        = 20
context_len     = 50
alpha           = 0.005
buffer_weight   = 0.8   # weight on TimesFM prediction when alarm fires

T_warmup        = context_len + 1
T_clean         = 20
attack_duration = 30
attack_start    = T_warmup + T_clean
T_trial         = T_warmup + T_clean + attack_duration
T_online_trial  = T_clean + attack_duration
threshold       = chi2.ppf(1 - alpha, df=M)

# Phase-aligned recording
period         = 2 * np.pi / omega
k_periods      = 1
T_record_start = int(attack_start - k_periods * period)

print(f"Exact period: {period:.3f} | T_record_start: {T_record_start} | attack_start: {attack_start}")
print(f"n={n_state} | M={M} | threshold={threshold:.3f}")
assert context_len <= T_warmup

# ============================================================
# STEP 1: ONE CLEAN RUN
# Purposes:
#   (a) record clean data for replay
#   (b) estimate TimesFM Sigma_fm
#   (c) estimate observer Sigma_obs
# ============================================================
np.random.seed(0)

T_init    = T_warmup + T_clean
x_init    = np.zeros((T_init + 1, n_state))
x_init[0] = x0
y_init    = np.zeros((T_init, M))

for t in range(T_init):
    x_init[t+1] = A @ x_init[t]
    y_init[t]   = C @ x_init[t] + sigma_n * np.random.randn(M)

# (a) Record for replay — phase aligned
recorded = y_init[T_record_start:T_record_start + T_clean].copy()

# (b) TimesFM Sigma — from clean window
res_fm_init = np.zeros((T_clean, M))
for t in range(T_clean):
    global_t = T_warmup + t
    start    = max(0, global_t - context_len)
    contexts = [y_init[start:global_t, m] for m in range(M)]
    pf, _    = model.forecast(horizon=1, inputs=contexts)
    res_fm_init[t] = y_init[global_t] - pf[:, 0]

Sigma_fm     = np.cov(res_fm_init.T)
Sigma_fm_inv = np.linalg.inv(Sigma_fm)

# (c) Observer Sigma — warm up observer then collect residuals over clean window
x_est_obs = np.zeros(n_state)
for t in range(T_warmup):
    innov     = y_init[t] - C @ x_est_obs
    x_est_obs = A @ x_est_obs + K_obs @ innov

res_obs_init = np.zeros((T_clean, M))
for t in range(T_clean):
    global_t  = T_warmup + t
    innov     = y_init[global_t] - C @ x_est_obs
    x_est_obs = A @ x_est_obs + K_obs @ innov
    res_obs_init[t] = innov

Sigma_obs     = np.cov(res_obs_init.T)
Sigma_obs_inv = np.linalg.inv(Sigma_obs)

print(f"Sigma_fm  diag: {np.diag(Sigma_fm)}")
print(f"Sigma_obs diag: {np.diag(Sigma_obs)}")

# ============================================================
# MONTE CARLO
# ============================================================
chi2_fm_att_trials  = np.zeros((N_trials, attack_duration))
chi2_obs_att_trials = np.zeros((N_trials, attack_duration))
chi2_fm_cln_trials  = np.zeros((N_trials, attack_duration))
chi2_obs_cln_trials = np.zeros((N_trials, attack_duration))

y_plot_att = None
y_plot_cln = None

print(f"\nMonte Carlo | N_trials={N_trials} | attack_duration={attack_duration} ...")

for trial in range(N_trials):
    np.random.seed(trial + 1)

    # --- Simulate: one state trajectory, two output vectors ---
    x_trial    = np.zeros((T_trial + 1, n_state))
    x_trial[0] = x0
    y_att      = np.zeros((T_trial, M))
    y_cln      = np.zeros((T_trial, M))

    # observer state for attacked and clean (static gain, warm up naturally)
    x_est_att  = np.zeros(n_state)
    x_est_cln  = np.zeros(n_state)

    res_obs_att = np.zeros((T_online_trial, M))
    res_obs_cln = np.zeros((T_online_trial, M))

    for t in range(T_trial):
        x_trial[t+1] = A @ x_trial[t]
        noise         = sigma_n * np.random.randn(M)
        y_cln[t]      = C @ x_trial[t] + noise

        if attack_start <= t < attack_start + attack_duration:
            replay_idx = (t - attack_start) % T_clean
            y_att[t]   = recorded[replay_idx]
        else:
            y_att[t]   = y_cln[t]

        # attacked observer update (static gain)
        innov_att  = y_att[t] - C @ x_est_att
        x_est_att  = A @ x_est_att + K_obs @ innov_att

        # clean observer update (static gain)
        innov_cln  = y_cln[t] - C @ x_est_cln
        x_est_cln  = A @ x_est_cln + K_obs @ innov_cln

        if t >= T_warmup:
            online_t = t - T_warmup
            res_obs_att[online_t] = innov_att
            res_obs_cln[online_t] = innov_cln

    # --- TimesFM residuals (buffer protection for attacked) ---
    res_fm_att = np.zeros((T_online_trial, M))
    res_fm_cln = np.zeros((T_online_trial, M))
    y_buffer   = y_att.copy()

    for t in range(T_warmup, T_trial):
        online_t = t - T_warmup
        start    = max(0, t - context_len)

        # attacked — use protected buffer
        contexts_att = [y_buffer[start:t, m] for m in range(M)]
        pf_att, _    = model.forecast(horizon=1, inputs=contexts_att)
        y_hat        = pf_att[:, 0]
        res_fm_att[online_t] = y_att[t] - y_hat

        # buffer protection: blend if alarm fires
        chi2_now = res_fm_att[online_t] @ Sigma_fm_inv @ res_fm_att[online_t]
        if chi2_now > threshold and t >= attack_start:
            y_buffer[t] = buffer_weight * y_hat + (1 - buffer_weight) * y_att[t]
        else:
            y_buffer[t] = y_att[t]

        # clean — no buffer needed
        contexts_cln = [y_cln[start:t, m] for m in range(M)]
        pf_cln, _    = model.forecast(horizon=1, inputs=contexts_cln)
        res_fm_cln[online_t] = y_cln[t] - pf_cln[:, 0]

    # --- Chi2 statistics ---
    chi2_fm_att  = np.array([res_fm_att[t]  @ Sigma_fm_inv  @ res_fm_att[t]  for t in range(T_online_trial)])
    chi2_fm_cln  = np.array([res_fm_cln[t]  @ Sigma_fm_inv  @ res_fm_cln[t]  for t in range(T_online_trial)])
    chi2_obs_att = np.array([res_obs_att[t] @ Sigma_obs_inv @ res_obs_att[t] for t in range(T_online_trial)])
    chi2_obs_cln = np.array([res_obs_cln[t] @ Sigma_obs_inv @ res_obs_cln[t] for t in range(T_online_trial)])

    # store only attack window
    chi2_fm_att_trials[trial]  = chi2_fm_att[T_clean:]
    chi2_fm_cln_trials[trial]  = chi2_fm_cln[T_clean:]
    chi2_obs_att_trials[trial] = chi2_obs_att[T_clean:]
    chi2_obs_cln_trials[trial] = chi2_obs_cln[T_clean:]

    if trial == 0:
        y_plot_att = y_att.copy()
        y_plot_cln = y_cln.copy()

    print(f"  Trial {trial+1}/{N_trials} done")

print("Done.")

# FAR and DR
FAR_fm  = np.mean(chi2_fm_cln_trials  > threshold)
FAR_obs = np.mean(chi2_obs_cln_trials > threshold)
DR_fm   = np.mean(np.any(chi2_fm_att_trials  > threshold, axis=1))
DR_obs  = np.mean(np.any(chi2_obs_att_trials > threshold, axis=1))
print(f"TimesFM  — FAR: {FAR_fm:.3f} | DR: {DR_fm:.3f}")
print(f"Observer — FAR: {FAR_obs:.3f} | DR: {DR_obs:.3f}")

# ============================================================
# PLOT
# ============================================================
fig = plt.figure(figsize=(14, 10.5))
gs  = gridspec.GridSpec(3, 2, figure=fig)

# --- Row 0: sensor trajectory ---
ax0 = fig.add_subplot(gs[0, :])
t_full = np.arange(len(y_plot_att))
ax0.plot(t_full, y_plot_att[:, 0], '-o', color='steelblue',
         lw=0.8, markersize=3, label='Sensor 1 (attacked)')
ax0.plot(t_full, y_plot_cln[:, 0], '--', color='gray',
         lw=0.8, alpha=0.6, label='Sensor 1 (clean)')
ax0.axvspan(T_record_start, T_record_start + T_clean,
            alpha=0.12, color='green', label='Recording window')
ax0.axvline(T_record_start, color='green', linestyle='--', lw=1.2)
ax0.axvline(T_record_start + T_clean, color='green', linestyle='--', lw=1.2)
ax0.axvline(attack_start, color='red', linestyle='--', lw=1.5, label='Replay start')
ax0.axvspan(attack_start, attack_start + attack_duration, alpha=0.08, color='red')
ax0.set_xlabel('Time step')
ax0.set_ylabel('Sensor 1 output')
ax0.set_title('Sensor trajectory — replay attack vs clean')
ax0.legend(fontsize=8)
ax0.grid(alpha=0.3)

def plot_band(ax, chi2_trials, color, title, t_offset=0):
    t_axis = np.arange(chi2_trials.shape[1]) + t_offset
    mean   = chi2_trials.mean(axis=0)
    std    = chi2_trials.std(axis=0)
    for i in range(chi2_trials.shape[0]):
        ax.plot(t_axis, chi2_trials[i], color=color, lw=0.5, alpha=0.3)
    ax.plot(t_axis, mean, '-+', color=color, lw=2.0, label='Mean')
    ax.fill_between(t_axis, mean - std, mean + std,
                    alpha=0.2, color=color, label='±1 std')
    ax.axhline(threshold, color='red', linestyle='--', lw=1.5,
               label=f'Threshold ($\\alpha$={alpha})')
    ax.set_ylabel('$\\chi^2$ statistic')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# --- Row 1: TimesFM ---
ax_fm_cln = fig.add_subplot(gs[1, 0])
ax_fm_att = fig.add_subplot(gs[1, 1])
plot_band(ax_fm_cln, chi2_fm_cln_trials, color='steelblue',
          title='TimesFM — no attack', t_offset=attack_start)
plot_band(ax_fm_att, chi2_fm_att_trials, color='steelblue',
          title=f'TimesFM — replay attack (duration={attack_duration})',
          t_offset=attack_start)
ax_fm_cln.set_xlabel('Time step')
ax_fm_att.set_xlabel('Time step in attack window')

# --- Row 2: Luenberger Observer ---
ax_obs_cln = fig.add_subplot(gs[2, 0])
ax_obs_att = fig.add_subplot(gs[2, 1])
plot_band(ax_obs_cln, chi2_obs_cln_trials, color='darkorange',
          title='Observer — no attack', t_offset=attack_start)
plot_band(ax_obs_att, chi2_obs_att_trials, color='darkorange',
          title=f'Observer — replay attack (duration={attack_duration})',
          t_offset=attack_start)
ax_obs_cln.set_xlabel('Time step')
ax_obs_att.set_xlabel('Time step in attack window')

# equal y-axis
y_max = max(ax_fm_cln.get_ylim()[1], ax_obs_cln.get_ylim()[1])
for ax in [ax_fm_cln, ax_obs_cln]:
    ax.set_ylim(0, y_max)

y_max = max(ax_fm_att.get_ylim()[1], ax_obs_att.get_ylim()[1])
for ax in [ax_fm_att, ax_obs_att]:
    ax.set_ylim(0, y_max)

plt.suptitle(
    f'Replay Attack | Number of trials={N_trials} | Context length={context_len} | FAR ($\\alpha$)={alpha}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('replay_luenberger.pdf', bbox_inches='tight')
plt.show()
