# ============================================================
# INSTALL & IMPORTS
# ============================================================
!pip install git+https://github.com/google-research/timesfm.git#egg=timesfm[torch] -q

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import chi2
from scipy.signal import place_poles
import timesfm

# ============================================================
# SYSTEM DEFINITION
# Single sensor C = [1, 0] — position only, system is observable
# ============================================================
np.random.seed(42)

omega   = 0.3
dt      = 1.0
n_state = 2
M       = 1          # single sensor
sigma_n = 0.01
x0      = np.array([1.0, 0.0])

A = np.array([
    [ np.cos(omega*dt),  np.sin(omega*dt)/omega],
    [-np.sin(omega*dt)*omega, np.cos(omega*dt)]
])

C = np.array([[1.0, 0.0]])   # shape (1, 2) — position only

# verify observability
O = np.vstack([C, C @ A])
assert np.linalg.matrix_rank(O) == n_state, "System not observable!"
print(f"System is observable. Rank={np.linalg.matrix_rank(O)}")

# ============================================================
# LUENBERGER OBSERVER — static gain via pole placement
# Observer: xhat[k+1] = A @ xhat[k] + K_obs @ (y[k] - C @ xhat[k])
# ============================================================
desired_poles = np.array([0.5 + 1e-1j, 0.5 - 1e-1j])   # conjugate pair for n=2
result        = place_poles(A.T, C.T, desired_poles)
K_obs         = result.gain_matrix.T   # shape (n_state, M)

print(f"Observer gain K_obs: {K_obs.flatten()}")

# ============================================================
# PARAMETERS
# ============================================================
N_trials        = 20
context_len     = 50
alpha           = 0.005
buffer_weight   = 1
attack_magnitude = 0.4   # tune this

T_warmup        = context_len + 1
T_clean         = 20
attack_duration = 20
attack_start    = T_warmup + T_clean
T_trial         = T_warmup + T_clean + attack_duration
T_online_trial  = T_clean + attack_duration
threshold       = chi2.ppf(1 - alpha, df=M)

assert context_len <= T_warmup
print(f"threshold={threshold:.3f} | attack_magnitude={attack_magnitude}")

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
# STEP 1: ONE CLEAN RUN
# Estimate Sigma_fm and Sigma_obs once
# ============================================================
np.random.seed(0)

T_init    = T_warmup + T_clean
x_init    = np.zeros((T_init + 1, n_state))
x_init[0] = x0
y_init    = np.zeros((T_init, M))

for t in range(T_init):
    x_init[t+1] = A @ x_init[t]
    y_init[t]   = C @ x_init[t] + sigma_n * np.random.randn(M)

# TimesFM Sigma
res_fm_init = np.zeros((T_clean, M))
for t in range(T_clean):
    global_t = T_warmup + t
    start    = max(0, global_t - context_len)
    contexts = [y_init[start:global_t, 0]]   # single channel
    pf, _    = model.forecast(horizon=1, inputs=contexts)
    res_fm_init[t] = y_init[global_t] - pf[:, 0]

Sigma_fm     = np.cov(res_fm_init.T).reshape(M, M)
Sigma_fm_inv = np.linalg.inv(Sigma_fm)

# Observer Sigma
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

Sigma_obs     = np.cov(res_obs_init.T).reshape(M, M)
Sigma_obs_inv = np.linalg.inv(Sigma_obs)

print(f"Sigma_fm:  {Sigma_fm[0,0]:.5f}")
print(f"Sigma_obs: {Sigma_obs[0,0]:.5f}")

# ============================================================
# MONTE CARLO
# Storage: state estimation error norm squared per trial per time step
# ============================================================
# unprotected observer state error
err_unprotected = np.zeros((N_trials, attack_duration))

# TimesFM-protected observer state error
err_protected   = np.zeros((N_trials, attack_duration))

# chi2 stats for plotting
chi2_fm_att_trials = np.zeros((N_trials, attack_duration))
chi2_fm_cln_trials = np.zeros((N_trials, attack_duration))

y_plot_att = None
y_plot_cln = None

print(f"\nMonte Carlo | N_trials={N_trials} | attack_magnitude={attack_magnitude} ...")

for trial in range(N_trials):
    np.random.seed(trial + 1)

    # --- Simulate true system ---
    x_trial    = np.zeros((T_trial + 1, n_state))
    x_trial[0] = x0
    y_cln      = np.zeros((T_trial, M))
    y_att      = np.zeros((T_trial, M))

    for t in range(T_trial):
        x_trial[t+1] = A @ x_trial[t]
        noise         = sigma_n * np.random.randn(M)
        y_cln[t]      = C @ x_trial[t] + noise
        if attack_start <= t < attack_start + attack_duration:
            y_att[t] = y_cln[t] + attack_magnitude*np.random.rand(1)   # step attack
        else:
            y_att[t] = y_cln[t]

    # --------------------------------------------------------
    # Unprotected observer — blindly uses y_att
    # --------------------------------------------------------
    x_est_unp = np.zeros(n_state)
    for t in range(T_trial):
        innov     = y_att[t] - C @ x_est_unp
        x_est_unp = A @ x_est_unp + K_obs @ innov

        if attack_start <= t < attack_start + attack_duration:
            k_att = t - attack_start
            err_unprotected[trial, k_att] = np.sum((x_est_unp - x_trial[t+1])**2)

    # --------------------------------------------------------
    # TimesFM detector + protected observer
    # When alarm fires: feed y_hat to observer instead of y_att
    #                   update buffer with weighted blend
    # --------------------------------------------------------
    x_est_prt = np.zeros(n_state)
    y_buffer  = y_att.copy()

    res_fm_att = np.zeros((T_online_trial, M))
    res_fm_cln = np.zeros((T_online_trial, M))

    for t in range(T_trial):
        online_t = t - T_warmup

        if t >= T_warmup:
            start = max(0, t - context_len)

            # TimesFM prediction on protected buffer
            contexts_att = [y_buffer[start:t, 0]]
            pf_att, _    = model.forecast(horizon=1, inputs=contexts_att)
            y_hat        = pf_att[:, 0]   # shape (1,)
            res_fm_att[online_t] = y_att[t] - y_hat

            # clean prediction
            contexts_cln = [y_cln[start:t, 0]]
            pf_cln, _    = model.forecast(horizon=1, inputs=contexts_cln)
            res_fm_cln[online_t] = y_cln[t] - pf_cln[:, 0]

            # check alarm
            chi2_now = float(res_fm_att[online_t] @ Sigma_fm_inv @ res_fm_att[online_t])

            if chi2_now > threshold and t >= attack_start:
                # alarm: feed TimesFM prediction to observer
                y_for_obs   = y_hat                    # use prediction not corrupted measurement
                y_buffer[t] = buffer_weight * y_hat + (1 - buffer_weight) * y_att[t]
            else:
                y_for_obs   = y_att[t]
                y_buffer[t] = y_att[t]
        else:
            # before warmup: no TimesFM yet, use y_att directly
            y_for_obs = y_att[t]
            if t >= T_warmup:
                res_fm_att[online_t] = 0.0
                res_fm_cln[online_t] = 0.0

        # protected observer update
        innov     = y_for_obs - C @ x_est_prt
        x_est_prt = A @ x_est_prt + K_obs @ innov

        if attack_start <= t < attack_start + attack_duration:
            k_att = t - attack_start
            err_protected[trial, k_att] = np.sum((x_est_prt - x_trial[t+1])**2)

    # store chi2 for attack window only
    chi2_fm_att_trials[trial] = np.array([
        res_fm_att[t] @ Sigma_fm_inv @ res_fm_att[t]
        for t in range(T_clean, T_online_trial)
    ])
    chi2_fm_cln_trials[trial] = np.array([
        res_fm_cln[t] @ Sigma_fm_inv @ res_fm_cln[t]
        for t in range(T_clean, T_online_trial)
    ])

    if trial == 0:
        y_plot_att = y_att.copy()
        y_plot_cln = y_cln.copy()

    print(f"  Trial {trial+1}/{N_trials} done")

print("Done.")

# FAR and DR
FAR_fm = np.mean(chi2_fm_cln_trials > threshold)
DR_fm  = np.mean(np.any(chi2_fm_att_trials > threshold, axis=1))
print(f"TimesFM — FAR: {FAR_fm:.3f} | DR: {DR_fm:.3f}")

# ============================================================
# PLOT
# ============================================================
fig = plt.figure(figsize=(14, 10.5))
gs  = gridspec.GridSpec(3, 2, figure=fig)

t_att_axis = np.arange(attack_duration) + attack_start

# --- Row 0: sensor trajectory (trial 0) ---
ax0 = fig.add_subplot(gs[0, :])
t_full = np.arange(T_trial)
ax0.plot(t_full, y_plot_att[:, 0], '-o', color='steelblue',
         lw=0.8, markersize=3, label='Sensor (attacked)')
ax0.plot(t_full, y_plot_cln[:, 0], '--', color='gray',
         lw=0.8, alpha=0.6, label='Sensor (clean)')
ax0.axvline(attack_start, color='red', linestyle='--', lw=1.5, label='Attack start')
ax0.axvline(attack_start + attack_duration, color='darkred',
            linestyle='--', lw=1.5, label='Attack end')
ax0.axvspan(attack_start, attack_start + attack_duration, alpha=0.08, color='red')
ax0.set_xlabel('Time step')
ax0.set_ylabel('Sensor output')
ax0.set_title(f'Sensor trajectory — step attack (magnitude={attack_magnitude})')
ax0.legend(fontsize=8)
ax0.grid(alpha=0.3)

# --- Row 1: State estimation error variance ---
ax1 = fig.add_subplot(gs[1, :])
var_unp = err_unprotected.var(axis=0)
var_prt = err_protected.var(axis=0)
mean_unp = err_unprotected.mean(axis=0)
mean_prt = err_protected.mean(axis=0)

ax1.plot(t_att_axis, mean_unp, '-o', color='crimson', lw=1.5,
         markersize=4, label='Unprotected observer (mean)')
ax1.fill_between(t_att_axis,
                 mean_unp - np.sqrt(var_unp),
                 mean_unp + np.sqrt(var_unp),
                 alpha=0.2, color='crimson', label='±1 std')
ax1.plot(t_att_axis, mean_prt, '-o', color='steelblue', lw=1.5,
         markersize=4, label='TimesFM-protected observer (mean)')
ax1.fill_between(t_att_axis,
                 mean_prt - np.sqrt(var_prt),
                 mean_prt + np.sqrt(var_prt),
                 alpha=0.2, color='steelblue', label='±1 std')
ax1.set_xlabel('Time step')
ax1.set_ylabel('$\|\\hat{x} - x\|^2$')
ax1.set_title('State estimation error — unprotected vs TimesFM-protected observer')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# --- Row 2: TimesFM chi2 statistic ---
def plot_band(ax, chi2_trials, color, title, t_offset=0):
    t_axis = np.arange(chi2_trials.shape[1]) + t_offset
    mean   = chi2_trials.mean(axis=0)
    std    = chi2_trials.std(axis=0)
    for i in range(chi2_trials.shape[0]):
        ax.plot(t_axis, chi2_trials[i], color=color, lw=0.4, alpha=0.2)
    ax.plot(t_axis, mean, color=color, lw=2.0, label='Mean')
    ax.fill_between(t_axis, mean - std, mean + std,
                    alpha=0.2, color=color, label='±1 std')
    ax.axhline(threshold, color='red', linestyle='--', lw=1.5,
               label=f'Threshold ($\\alpha$={alpha})')
    ax.set_ylabel('$\\chi^2$ statistic')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

ax_fm_cln = fig.add_subplot(gs[2, 0])
ax_fm_att = fig.add_subplot(gs[2, 1])
plot_band(ax_fm_cln, chi2_fm_cln_trials, color='steelblue',
          title='TimesFM — no attack', t_offset=attack_start)
plot_band(ax_fm_att, chi2_fm_att_trials, color='steelblue',
          title=f'TimesFM — step attack (magnitude={attack_magnitude})',
          t_offset=attack_start)
ax_fm_cln.set_xlabel('Time step')
ax_fm_att.set_xlabel('Time step in attack window')

plt.suptitle(
    f'Attack Mitigation | N_trials={N_trials} | magnitude={attack_magnitude} | $\\alpha$={alpha}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('attack_mitigation.pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))

t_att_axis = np.arange(attack_duration) + attack_start

mean_unp = err_unprotected.mean(axis=0)
std_unp  = err_unprotected.std(axis=0)
mean_prt = err_protected.mean(axis=0)
std_prt  = err_protected.std(axis=0)

ax.plot(t_att_axis, mean_unp, '-o', color='crimson', lw=1.5,
        markersize=4, label='Unprotected observer')
ax.fill_between(t_att_axis, mean_unp - std_unp, mean_unp + std_unp,
                alpha=0.2, color='crimson', label='±1 std')

ax.plot(t_att_axis, mean_prt, '-o', color='steelblue', lw=1.5,
        markersize=4, label='TimesFM-protected observer')
ax.fill_between(t_att_axis, mean_prt - std_prt, mean_prt + std_prt,
                alpha=0.2, color='steelblue', label='±1 std')

ax.axvline(attack_start, color='red', linestyle='--', lw=1.5, label='Attack start')
# ax.axvspan(attack_start, attack_start + attack_duration, alpha=0.08, color='red')
ax.set_xlabel('Time step in attack window')
ax.set_ylabel('Norm of state estimation error')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle(
    f'Attack Mitigation | Number of trials={N_trials} | Context length={context_len} | FAR ($\\alpha$)={alpha}',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('attack_mitigation.pdf', bbox_inches='tight')
plt.show()