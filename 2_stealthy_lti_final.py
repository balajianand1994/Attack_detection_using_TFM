!pip install git+https://github.com/google-research/timesfm.git#egg=timesfm[torch] -q
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import chi2
from scipy.signal import place_poles
import timesfm

np.random.seed(42)

omega   = 0.3
dt      = 1.0
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
N_trials        = 20
context_len     = 50
alpha_det       = 0.005
buffer_weight   = 0.8    # weight on predicted value when alarm fires

Q_kf = np.zeros((2, 2))
R_kf = np.eye(M) * sigma_n**2

T_warmup     = context_len + 1
T_clean      = 30
attack_start = T_warmup + T_clean
threshold    = chi2.ppf(1 - alpha_det, df=M)

assert context_len <= T_warmup


# ============================================================
# LUENBERGER OBSERVER via pole placement
# ============================================================
# Place observer poles inside unit circle (faster than system poles)
desired_poles = np.array([0.5 + 1e-1j, 0.5 - 1e-1j])   # conjugate pair for n=2
result        = place_poles(A.T, C.T, desired_poles)
K_obs         = result.gain_matrix.T   # shape (2, M)

print(f"Observer poles: {desired_poles}")
print(f"Observer gain K_obs:\n{K_obs}")

# ============================================================
# OPTIMAL ATTACK DESIGN
# ============================================================
attack_duration = 20
T_att           = attack_duration       # attack horizon
eta             = 0.03*threshold             # attack energy budget = detector threshold
q               = np.array([1.0, 0.0]) # weight on first state

# Precompute c_k = K_obs.T @ (A^{T-1-k}).T @ q for k=0,...,T-1
c_k_list = []
for k in range(T_att):
    Apow     = np.linalg.matrix_power(A, T_att - 1 - k)
    c_k      = K_obs.T @ Apow.T @ q    # shape (M,)
    c_k_list.append(c_k)

# ============================================================
# STEP 1: ONE CLEAN RUN — estimate Sigma for TimesFM and observer
# ============================================================
np.random.seed(0)

T_init     = T_warmup + T_clean
x_init     = np.zeros((T_init + 1, 2))
x_init[0]  = x0
y_init     = np.zeros((T_init, M))

for t in range(T_init):
    x_init[t+1] = A @ x_init[t]
    y_init[t]   = C @ x_init[t] + sigma_n * np.random.randn(M)

# TimesFM Sigma
res_fm_init = np.zeros((T_clean, M))
for t in range(T_clean):
    global_t = T_warmup + t
    start    = max(0, global_t - context_len)
    contexts = [y_init[start:global_t, m] for m in range(M)]
    point_forecast, _ = model.forecast(horizon=1, inputs=contexts)
    res_fm_init[t] = y_init[global_t] - point_forecast[:, 0]

Sigma_fm     = np.cov(res_fm_init.T)
Sigma_fm_inv = np.linalg.inv(Sigma_fm)

# Observer Sigma — warm up observer on clean run, collect residuals
x_est_obs = np.zeros(2)
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

print(f"Sigma_fm diag:  {np.diag(Sigma_fm)}")
print(f"Sigma_obs diag: {np.diag(Sigma_obs)}")

# ============================================================
# MONTE CARLO
# ============================================================
T_trial        = T_warmup + T_clean + attack_duration
T_online_trial = T_clean + attack_duration

chi2_fm_att_trials  = np.zeros((N_trials, attack_duration))
chi2_obs_att_trials = np.zeros((N_trials, attack_duration))
chi2_fm_cln_trials  = np.zeros((N_trials, attack_duration))
chi2_obs_cln_trials = np.zeros((N_trials, attack_duration))

y_plot_att = None
y_plot_cln = None

print(f"\nOptimal attack | duration={attack_duration} | eta={eta:.3f} ...")

for trial in range(N_trials):
    np.random.seed(trial + 1)

    # --- Simulate: one state trajectory, two output vectors ---
    x_trial = np.zeros((T_trial + 1, 2))
    x_trial[0] = x0
    y_cln   = np.zeros((T_trial, M))

    for t in range(T_trial):
        x_trial[t+1] = A @ x_trial[t]
        y_cln[t]      = C @ x_trial[t] + sigma_n * np.random.randn(M)

    # --- Design optimal attack using observer residuals ---
    # epsilon_k = x_hat_attacked - x_hat_clean, starts at 0
    epsilon   = np.zeros(2)
    y_att     = y_cln.copy()

    for t in range(T_trial):
        if attack_start <= t < attack_start + attack_duration:
            k_att = t - attack_start
            c_k   = c_k_list[k_att]

            # optimal delta_k: maximize c_k^T delta_k s.t. delta_k^T Sigma^{-1} delta_k <= eta
            Sigma_c  = Sigma_obs @ c_k
            denom    = np.sqrt(c_k @ Sigma_obs @ c_k)
            if denom > 1e-10:
                delta_k = np.sqrt(eta) * Sigma_c / denom
            else:
                delta_k = np.zeros(M)

            # attack = delta_k + C @ epsilon (to account for observer deviation)
            a_t      = delta_k + C @ epsilon
            y_att[t] = y_cln[t] + a_t

            # update epsilon: epsilon_{k+1} = A @ epsilon + K_obs @ delta_k
            epsilon  = (A - K_obs @ C) @ epsilon + K_obs @ a_t
        else:
            epsilon = (A - K_obs @ C) @ epsilon

    # --- TimesFM residuals (with buffer protection) ---
    res_fm_att = np.zeros((T_online_trial, M))
    res_fm_cln = np.zeros((T_online_trial, M))

    # maintain a protected buffer for attacked trajectory
    y_buffer = y_att.copy()

    for t in range(T_warmup, T_trial):
        online_t = t - T_warmup
        start    = max(0, t - context_len)

        # --- attacked: use protected buffer ---
        contexts_att = [y_buffer[start:t, m] for m in range(M)]
        pf_att, _    = model.forecast(horizon=1, inputs=contexts_att)
        y_hat        = pf_att[:, 0]
        res_fm_att[online_t] = y_att[t] - y_hat

        # check if alarm fires — if so, update buffer with weighted prediction
        chi2_now = res_fm_att[online_t] @ Sigma_fm_inv @ res_fm_att[online_t]
        if chi2_now > threshold and t >= attack_start:
            y_buffer[t] = buffer_weight * y_hat + (1 - buffer_weight) * y_att[t]
        else:
            y_buffer[t] = y_att[t]

        # --- clean ---
        contexts_cln = [y_cln[start:t, m] for m in range(M)]
        pf_cln, _    = model.forecast(horizon=1, inputs=contexts_cln)
        res_fm_cln[online_t] = y_cln[t] - pf_cln[:, 0]

    # --- Observer residuals ---
    res_obs_att = np.zeros((T_online_trial, M))
    res_obs_cln = np.zeros((T_online_trial, M))

    x_est_att = np.zeros(2);  x_est_cln = np.zeros(2)

    for t in range(T_trial):
        innov_att     = y_att[t] - C @ x_est_att
        x_est_att     = A @ x_est_att + K_obs @ innov_att

        innov_cln     = y_cln[t] - C @ x_est_cln
        x_est_cln     = A @ x_est_cln + K_obs @ innov_cln

        if t >= T_warmup:
            online_t = t - T_warmup
            res_obs_att[online_t] = innov_att
            res_obs_cln[online_t] = innov_cln

    # --- Chi2 statistics ---
    chi2_fm_att  = np.array([res_fm_att[t]  @ Sigma_fm_inv  @ res_fm_att[t]  for t in range(T_online_trial)])
    chi2_fm_cln  = np.array([res_fm_cln[t]  @ Sigma_fm_inv  @ res_fm_cln[t]  for t in range(T_online_trial)])
    chi2_obs_att = np.array([res_obs_att[t] @ Sigma_obs_inv @ res_obs_att[t] for t in range(T_online_trial)])
    chi2_obs_cln = np.array([res_obs_cln[t] @ Sigma_obs_inv @ res_obs_cln[t] for t in range(T_online_trial)])

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
print(f"TimesFM  — FAR: {FAR_fm:.3f}  | DR: {DR_fm:.3f}")
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
ax0.axvline(attack_start, color='red', linestyle='--', lw=1.5, label='Attack start')
ax0.axvline(attack_start + attack_duration, color='darkred', linestyle='--', lw=1.5, label='Attack end')
ax0.axvspan(attack_start, attack_start + attack_duration, alpha=0.08, color='red')
ax0.set_xlabel('Time step')
ax0.set_ylabel('Sensor 1 output')
ax0.set_title('Sensor trajectory — stealthy attack vs clean')
ax0.legend(fontsize=8)
ax0.grid(alpha=0.3)

def plot_band(ax, chi2_trials, color, title, t_offset=0):
    t_axis = np.arange(chi2_trials.shape[1]) + t_offset
    mean   = chi2_trials.mean(axis=0)
    std    = chi2_trials.std(axis=0)
    for trial in range(chi2_trials.shape[0]):
        ax.plot(t_axis, chi2_trials[trial], color=color, lw=0.5, alpha=0.3)
    ax.plot(t_axis, mean, '-+', color=color, lw=2.0, label='Mean')
    ax.fill_between(t_axis, mean - std, mean + std,
                    alpha=0.2, color=color, label='±1 std')
    ax.axhline(threshold, color='red', linestyle='--', lw=1.5,
               label=f'Threshold ($\\alpha$={alpha_det})')
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
          title=f'TimesFM — optimal attack (duration={attack_duration})', t_offset=attack_start)
ax_fm_cln.set_xlabel('Time step')
ax_fm_att.set_xlabel('Time step in attack window')

# --- Row 2: Observer ---
ax_obs_cln = fig.add_subplot(gs[2, 0])
ax_obs_att = fig.add_subplot(gs[2, 1])
plot_band(ax_obs_cln, chi2_obs_cln_trials, color='darkorange',
          title='Observer — no attack', t_offset=attack_start)
plot_band(ax_obs_att, chi2_obs_att_trials, color='darkorange',
          title=f'Observer — optimal attack (duration={attack_duration})', t_offset=attack_start)
ax_obs_cln.set_xlabel('Time step')
ax_obs_att.set_xlabel('Time step in attack window')

# Equal y-axis
y_max = max(ax_fm_cln.get_ylim()[1], ax_obs_cln.get_ylim()[1],ax_obs_att.get_ylim()[1])
for ax in [ax_fm_cln, ax_obs_cln, ax_obs_att]:
    ax.set_ylim(0, y_max)

y_max = max(ax_fm_att.get_ylim()[1],ax_obs_att.get_ylim()[1])
for ax in [ax_fm_att]:
    ax.set_ylim(0, y_max)

plt.suptitle(f'Stealthy attacks | Number of trials={N_trials} | Context length={context_len} | FAR ($\\alpha$)={alpha_det} | $\\Delta \\tau_p$ = {eta:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('optimal_attack_chi2.pdf', bbox_inches='tight')
plt.show()
