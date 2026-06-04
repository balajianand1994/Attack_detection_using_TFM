# ============================================================
# IMPORTS & SYSTEM DEFINITION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import chi2
import pickle

np.random.seed(42)

# ============================================================
# LOAD TIMESFM
# ============================================================
!pip install git+https://github.com/google-research/timesfm.git#egg=timesfm[torch] -q
import timesfm
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(
    timesfm.ForecastConfig(max_context=512, max_horizon=1, normalize_inputs=True)
)
print("Model loaded.")

# ============================================================
# IEEE 14-BUS SYSTEM DEFINITION
# ============================================================
np.random.seed(42)

Nbus = 14
branches = np.array([
    [1,2],[1,5],[2,3],[2,4],[2,5],[3,4],[4,5],[4,7],[4,9],
    [5,6],[6,11],[6,12],[6,13],[7,8],[7,9],[9,10],[9,14],
    [10,11],[12,13],[13,14]
]) - 1

Nline   = len(branches)
n_state = (Nbus - 1) + Nbus
m_meas  = Nbus + Nline
dt      = 0.02

Bline = 4.0 * np.ones(Nline)
Bline[0]  = 0.05917;  Bline[1]  = 0.22304;  Bline[2]  = 0.19797
Bline[3]  = 0.17632;  Bline[4]  = 0.17388;  Bline[5]  = 0.17103
Bline[6]  = 0.04211;  Bline[7]  = 0.20912;  Bline[8]  = 0.55618
Bline[9]  = 0.25202;  Bline[10] = 0.25202;  Bline[11] = 0.25581
Bline[12] = 0.13027;  Bline[13] = 0.17615;  Bline[14] = 0.11001
Bline[15] = 0.08450;  Bline[16] = 0.27038;  Bline[17] = 0.19207
Bline[18] = 0.19988;  Bline[19] = 0.34802

M_bus = [1.06,1.045,1.01,1.019,1.02,1.07,1.062,1.09,1.056,1.051,1.057,1.055,1.05,1.036]
D_bus = [0,4.98,12.72,10.33,8.78,14.22,13.37,13.36,14.94,15.1,14.79,15.07,15.16,16.04]

sigma_omega = 0.035
sigma_flow  = 0.050
R_noise = np.diag(np.concatenate([
    sigma_omega**2 * np.ones(Nbus),
    sigma_flow**2  * np.ones(Nline)
]))
Q_noise = 1e-5 * np.eye(n_state)
Rhalf   = np.linalg.cholesky(R_noise)

# ============================================================
# SYSTEM FUNCTIONS
# ============================================================
def f_swing(x):
    theta = x[:Nbus-1]
    omega = x[Nbus-1:]
    delta = np.concatenate([[0.0], theta])
    Pe    = np.zeros(Nbus)
    for ell, (i, j) in enumerate(branches):
        flow   = Bline[ell] * np.sin(delta[i] - delta[j])
        Pe[i] += flow
        Pe[j] -= flow
    omega_dot = (-Pe - D_bus * omega) / M_bus
    theta_dot = omega[1:] - omega[0]
    return np.concatenate([theta + dt * theta_dot,
                           omega + dt * omega_dot])

def h_meas(x):
    theta = x[:Nbus-1]
    omega = x[Nbus-1:]
    delta = np.concatenate([[0.0], theta])
    flows = np.array([
        np.sin(delta[branches[ell,0]] - delta[branches[ell,1]])
        for ell in range(Nline)
    ])
    return np.concatenate([omega, flows])

def numerical_jacobian(fun, x, eps=1e-6):
    fx = fun(x)
    J  = np.zeros((len(fx), len(x)))
    for i in range(len(x)):
        dx     = np.zeros(len(x))
        dx[i]  = eps * max(1.0, abs(x[i]))
        J[:,i] = (fun(x + dx) - fun(x - dx)) / (2.0 * dx[i])
    return J

# ============================================================
# PARAMETERS
# ============================================================
N_trials        = 10
context_len     = 50
alpha           = 0.001
buffer_weight   = 1.0

T_warmup        = context_len + 10
T_clean         = 10     # record window for replay
attack_duration = 10
attack_start    = T_warmup + T_clean
T_trial         = T_warmup + T_clean + attack_duration
T_online_trial  = T_clean + attack_duration

threshold       = chi2.ppf(1 - alpha, df=m_meas)

assert context_len <= T_warmup
print(f"n={n_state} | m={m_meas} | threshold={threshold:.3f}")

# ============================================================
# LOAD SIGMA_FM
# ============================================================
sigma_path   = 'sigma_fm.npy'
Sigma_fm     = np.load(sigma_path)
Sigma_fm_inv = np.linalg.inv(Sigma_fm)
print(f"Sigma_fm loaded | diag mean: {np.diag(Sigma_fm).mean():.4f}")

# ============================================================
# RECORD CLEAN DATA FOR REPLAY — one clean run with seed 0
# Records T_clean steps just before attack_start
# ============================================================
np.random.seed(0)

T_record      = T_warmup + T_clean   # simulate up to attack_start
x_rec         = np.zeros((T_record + 1, n_state))
x_rec[0]      = np.concatenate([
    0.08 * np.random.randn(Nbus - 1),
    0.02 * np.random.randn(Nbus)
])
y_rec         = np.zeros((T_record, m_meas))

for t in range(T_record):
    x_rec[t+1] = f_swing(x_rec[t])
    y_rec[t]   = h_meas(x_rec[t]) + Rhalf @ np.random.randn(m_meas)

# record last T_clean steps — just before attack_start
recorded = y_rec[T_warmup:T_warmup + T_clean].copy()   # shape (T_clean, m_meas)
print(f"Recorded shape: {recorded.shape}")

# ============================================================
# VERIFY EKF INNOVATIONS ARE GAUSSIAN — Q-Q PLOT
# ============================================================
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(0)

# Short clean run to collect innovations
T_verify   = 500
x_ver      = np.zeros((T_verify + 1, n_state))
x_ver[0]   = np.concatenate([0.08 * np.random.randn(Nbus-1),
                               0.02 * np.random.randn(Nbus)])
xhat_ver   = np.zeros((T_verify + 1, n_state))
xhat_ver[0] = x_ver[0] + 0.01 * np.random.randn(n_state)
P_ver      = 0.01 * np.eye(n_state)

innovations = np.zeros((T_verify, m_meas))

for t in range(T_verify):
    x_ver[t+1]  = f_swing(x_ver[t])
    y_ver       = h_meas(x_ver[t]) + Rhalf @ np.random.randn(m_meas)

    xpred       = f_swing(xhat_ver[t])
    Fk          = numerical_jacobian(f_swing, xhat_ver[t])
    Ppred       = Fk @ P_ver @ Fk.T + Q_noise
    Ppred       = 0.5 * (Ppred + Ppred.T)
    ypred       = h_meas(xpred)
    Hk          = numerical_jacobian(h_meas, xpred)
    rk          = y_ver - ypred
    Sk          = Hk @ Ppred @ Hk.T + R_noise
    Sk          = 0.5 * (Sk + Sk.T)
    Kk          = Ppred @ Hk.T @ np.linalg.inv(Sk)
    xhat_ver[t+1] = xpred + Kk @ rk
    P_ver       = (np.eye(n_state) - Kk @ Hk) @ Ppred @ \
                  (np.eye(n_state) - Kk @ Hk).T + Kk @ R_noise @ Kk.T
    P_ver       = 0.5 * (P_ver + P_ver.T) + 1e-12 * np.eye(n_state)

    innovations[t] = rk

# Drop warmup steps
warmup_steps = 100
innov_steady = innovations[warmup_steps:]

# Normalize innovations: r_normalized[t,i] = r[t,i] / std(r[:,i])
innov_norm = (innov_steady - innov_steady.mean(axis=0)) / innov_steady.std(axis=0)

# --- Q-Q plots: pick a representative subset of channels ---
# Plot all 34 channels in a grid
n_cols = 6
n_rows = int(np.ceil(m_meas / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
axes = axes.flatten()

for ch in range(m_meas):
    ax = axes[ch]
    stats.probplot(innov_norm[:, ch], dist='norm', plot=ax)
    ax.set_title(f'Ch {ch+1}', fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_lines()[0].set(markersize=2, alpha=0.5)

# hide unused axes
for ch in range(m_meas, len(axes)):
    axes[ch].set_visible(False)

plt.suptitle(f'Q-Q plots of normalized EKF innovations (T={T_verify - warmup_steps} steps)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('qq_innovations.pdf', bbox_inches='tight')
plt.show()

# Also print Shapiro-Wilk test for each channel
print("\nShapiro-Wilk normality test (p > 0.05 suggests Gaussian):")
for ch in range(m_meas):
    stat, p = stats.shapiro(innov_norm[:, ch])
    flag = '✓' if p > 0.05 else '✗'
    print(f"  Ch {ch+1:2d}: W={stat:.4f}, p={p:.4f} {flag}")

# ============================================================
# VERIFY TIMESFM RESIDUALS ARE GAUSSIAN — Q-Q PLOT
# ============================================================

# Reuse y_ver from the EKF verification run (same clean trajectory)
# Run TimesFM over the steady-state window (after warmup)

T_fm_start  = warmup_steps   # skip warmup
T_fm_end    = T_verify
n_fm_steps  = T_fm_end - T_fm_start

res_fm_ver  = np.zeros((n_fm_steps, m_meas))

print("Running TimesFM on clean verification trajectory ...")
for t in range(T_fm_start, T_fm_end):
    start    = max(0, t - context_len)
    contexts = [innov_steady[start:t, ch] if t > T_fm_start
                else [0.0]
                for ch in range(m_meas)]

    # use y_ver trajectory — need to reconstruct it
    # easier: re-simulate y_ver
    pass

# Actually easier to re-simulate y_ver cleanly
np.random.seed(0)
x_ver2     = np.zeros((T_verify + 1, n_state))
x_ver2[0]  = np.concatenate([0.08 * np.random.randn(Nbus-1),
                               0.02 * np.random.randn(Nbus)])
y_ver2     = np.zeros((T_verify, m_meas))

for t in range(T_verify):
    x_ver2[t+1] = f_swing(x_ver2[t])
    y_ver2[t]   = h_meas(x_ver2[t]) + Rhalf @ np.random.randn(m_meas)

res_fm_ver = np.zeros((n_fm_steps, m_meas))

for t in range(T_fm_start, T_fm_end):
    idx      = t - T_fm_start
    start    = max(0, t - context_len)
    contexts = [y_ver2[start:t, ch] for ch in range(m_meas)]
    pf, _    = model.forecast(horizon=1, inputs=contexts)
    res_fm_ver[idx] = y_ver2[t] - pf[:, 0]

    if idx % 50 == 0:
        print(f"  step {idx}/{n_fm_steps}")

print("Done.")

# Normalize
res_fm_norm = (res_fm_ver - res_fm_ver.mean(axis=0)) / res_fm_ver.std(axis=0)

# --- Q-Q plots ---
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
axes = axes.flatten()

for ch in range(m_meas):
    ax = axes[ch]
    stats.probplot(res_fm_norm[:, ch], dist='norm', plot=ax)
    ax.set_title(f'Ch {ch+1}', fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_lines()[0].set(markersize=2, alpha=0.5)

for ch in range(m_meas, len(axes)):
    axes[ch].set_visible(False)

plt.suptitle(f'Q-Q plots of normalized TimesFM residuals (T={n_fm_steps} steps)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('qq_timesfm.pdf', bbox_inches='tight')
plt.show()

# Shapiro-Wilk
print("\nShapiro-Wilk normality test — TimesFM residuals:")
for ch in range(m_meas):
    stat, p = stats.shapiro(res_fm_norm[:, ch])
    flag = '✓' if p > 0.05 else '✗'
    print(f"  Ch {ch+1:2d}: W={stat:.4f}, p={p:.4f} {flag}")

# ============================================================
# VERIFY TIMESFM RESIDUALS ARE GAUSSIAN — Q-Q PLOT
# ============================================================

# Reuse y_ver from the EKF verification run (same clean trajectory)
# Run TimesFM over the steady-state window (after warmup)

T_fm_start  = warmup_steps   # skip warmup
T_fm_end    = T_verify
n_fm_steps  = T_fm_end - T_fm_start

res_fm_ver  = np.zeros((n_fm_steps, m_meas))

print("Running TimesFM on clean verification trajectory ...")
for t in range(T_fm_start, T_fm_end):
    start    = max(0, t - context_len)
    contexts = [innov_steady[start:t, ch] if t > T_fm_start
                else [0.0]
                for ch in range(m_meas)]

    # use y_ver trajectory — need to reconstruct it
    # easier: re-simulate y_ver
    pass

# Actually easier to re-simulate y_ver cleanly
np.random.seed(0)
x_ver2     = np.zeros((T_verify + 1, n_state))
x_ver2[0]  = np.concatenate([0.08 * np.random.randn(Nbus-1),
                               0.02 * np.random.randn(Nbus)])
y_ver2     = np.zeros((T_verify, m_meas))

for t in range(T_verify):
    x_ver2[t+1] = f_swing(x_ver2[t])
    y_ver2[t]   = h_meas(x_ver2[t]) + Rhalf @ np.random.randn(m_meas)

res_fm_ver = np.zeros((n_fm_steps, m_meas))

for t in range(T_fm_start, T_fm_end):
    idx      = t - T_fm_start
    start    = max(0, t - context_len)
    contexts = [y_ver2[start:t, ch] for ch in range(m_meas)]
    pf, _    = model.forecast(horizon=1, inputs=contexts)
    res_fm_ver[idx] = y_ver2[t] - pf[:, 0]

    if idx % 50 == 0:
        print(f"  step {idx}/{n_fm_steps}")

print("Done.")

# Normalize
res_fm_norm = (res_fm_ver - res_fm_ver.mean(axis=0)) / res_fm_ver.std(axis=0)

# --- Q-Q plots ---
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
axes = axes.flatten()

for ch in range(m_meas):
    ax = axes[ch]
    stats.probplot(res_fm_norm[:, ch], dist='norm', plot=ax)
    ax.set_title(f'Ch {ch+1}', fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_lines()[0].set(markersize=2, alpha=0.5)

for ch in range(m_meas, len(axes)):
    axes[ch].set_visible(False)

plt.suptitle(f'Q-Q plots of normalized TimesFM residuals (T={n_fm_steps} steps)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('qq_timesfm.pdf', bbox_inches='tight')
plt.show()

# Shapiro-Wilk
print("\nShapiro-Wilk normality test — TimesFM residuals:")
for ch in range(m_meas):
    stat, p = stats.shapiro(res_fm_norm[:, ch])
    flag = '✓' if p > 0.05 else '✗'
    print(f"  Ch {ch+1:2d}: W={stat:.4f}, p={p:.4f} {flag}")

# ============================================================
# MONTE CARLO
# ============================================================
chi2_fm_att_trials  = np.zeros((N_trials, attack_duration))
chi2_ekf_att_trials = np.zeros((N_trials, attack_duration))
chi2_fm_cln_trials  = np.zeros((N_trials, attack_duration))
chi2_ekf_cln_trials = np.zeros((N_trials, attack_duration))

y_plot_att = None
y_plot_cln = None

print(f"\nMonte Carlo | N_trials={N_trials} | attack_duration={attack_duration} ...")

for trial in range(N_trials):
    np.random.seed(trial + 1)
    print(f"\n  Trial {trial+1}/{N_trials} ...")

    # --------------------------------------------------------
    # Step A: simulate true system with new noise realization
    # --------------------------------------------------------
    theta0_t    = 0.08 * np.random.randn(Nbus - 1)
    omega0_t    = 0.02 * np.random.randn(Nbus)
    x_trial     = np.zeros((T_trial + 1, n_state))
    x_trial[0]  = np.concatenate([theta0_t, omega0_t])

    xhat_nom_t    = np.zeros((T_trial + 1, n_state))
    xhat_nom_t[0] = x_trial[0] + 0.01 * np.random.randn(n_state)

    P_t = 0.01 * np.eye(n_state)

    y_cln     = np.zeros((T_trial, m_meas))
    S_store_t = np.zeros((T_trial, m_meas, m_meas))

    # --------------------------------------------------------
    # Step B: nominal EKF run — store S_k per trial (for chi2)
    # No attack design needed — replay attack is pre-fixed
    # --------------------------------------------------------
    for t in range(T_trial):
        x_trial[t+1] = f_swing(x_trial[t])
        y_cln[t]     = h_meas(x_trial[t]) + Rhalf @ np.random.randn(m_meas)

        xpred        = f_swing(xhat_nom_t[t])
        Fk           = numerical_jacobian(f_swing, xhat_nom_t[t])
        Ppred        = Fk @ P_t @ Fk.T + Q_noise
        Ppred        = 0.5 * (Ppred + Ppred.T)
        ypred        = h_meas(xpred)
        Hk           = numerical_jacobian(h_meas, xpred)
        rk           = y_cln[t] - ypred
        Sk           = Hk @ Ppred @ Hk.T + R_noise
        Sk           = 0.5 * (Sk + Sk.T)
        Kk           = Ppred @ Hk.T @ np.linalg.inv(Sk)
        xhat_nom_t[t+1] = xpred + Kk @ rk
        P_t          = (np.eye(n_state) - Kk @ Hk) @ Ppred @ (np.eye(n_state) - Kk @ Hk).T \
                       + Kk @ R_noise @ Kk.T
        P_t          = 0.5 * (P_t + P_t.T) + 1e-12 * np.eye(n_state)

        S_store_t[t] = Sk   # store for chi2 statistic

    print(f"    Nominal EKF done.")

    # --------------------------------------------------------
    # Step C: build y_att using replay attack
    # All sensors replayed — no attack design needed
    # --------------------------------------------------------
    y_att = y_cln.copy()
    for t in range(T_trial):
        if attack_start <= t < attack_start + attack_duration:
            replay_idx = (t - attack_start) % T_clean
            y_att[t]   = recorded[replay_idx]   # replay all sensors

    # --------------------------------------------------------
    # Step D: attacked and clean EKF forward pass
    # Uses stored K_k (nominal gain) — same as stealthy attack code
    # --------------------------------------------------------
    xhat_att    = np.zeros((T_trial + 1, n_state))
    xhat_cln    = np.zeros((T_trial + 1, n_state))
    xhat_att[0] = xhat_nom_t[0].copy()
    xhat_cln[0] = xhat_nom_t[0].copy()

    res_ekf_att = np.zeros((T_online_trial, m_meas))
    res_ekf_cln = np.zeros((T_online_trial, m_meas))

    # reuse nominal EKF state for gain — re-run to get K_k
    # (we need K_k for the update; re-extract from stored quantities)
    xhat_att2    = np.zeros((T_trial + 1, n_state))
    xhat_cln2    = np.zeros((T_trial + 1, n_state))
    xhat_att2[0] = xhat_nom_t[0].copy()
    xhat_cln2[0] = xhat_nom_t[0].copy()
    P_att2 = 0.01 * np.eye(n_state)
    P_cln2 = 0.01 * np.eye(n_state)

    for t in range(T_trial):
        # attacked EKF
        xpred_att  = f_swing(xhat_att2[t])
        Fk_att     = numerical_jacobian(f_swing, xhat_att2[t])
        Ppred_att  = Fk_att @ P_att2 @ Fk_att.T + Q_noise
        Ppred_att  = 0.5 * (Ppred_att + Ppred_att.T)
        ypred_att  = h_meas(xpred_att)
        Hk_att     = numerical_jacobian(h_meas, xpred_att)
        r_att      = y_att[t] - ypred_att
        Sk_att     = Hk_att @ Ppred_att @ Hk_att.T + R_noise
        Sk_att     = 0.5 * (Sk_att + Sk_att.T)
        Kk_att     = Ppred_att @ Hk_att.T @ np.linalg.inv(Sk_att)
        xhat_att2[t+1] = xpred_att + Kk_att @ r_att
        P_att2     = (np.eye(n_state) - Kk_att @ Hk_att) @ Ppred_att @ \
                     (np.eye(n_state) - Kk_att @ Hk_att).T + Kk_att @ R_noise @ Kk_att.T
        P_att2     = 0.5 * (P_att2 + P_att2.T) + 1e-12 * np.eye(n_state)

        # clean EKF
        xpred_cln  = f_swing(xhat_cln2[t])
        Fk_cln     = numerical_jacobian(f_swing, xhat_cln2[t])
        Ppred_cln  = Fk_cln @ P_cln2 @ Fk_cln.T + Q_noise
        Ppred_cln  = 0.5 * (Ppred_cln + Ppred_cln.T)
        ypred_cln  = h_meas(xpred_cln)
        Hk_cln     = numerical_jacobian(h_meas, xpred_cln)
        r_cln      = y_cln[t] - ypred_cln
        Sk_cln     = Hk_cln @ Ppred_cln @ Hk_cln.T + R_noise
        Sk_cln     = 0.5 * (Sk_cln + Sk_cln.T)
        Kk_cln     = Ppred_cln @ Hk_cln.T @ np.linalg.inv(Sk_cln)
        xhat_cln2[t+1] = xpred_cln + Kk_cln @ r_cln
        P_cln2     = (np.eye(n_state) - Kk_cln @ Hk_cln) @ Ppred_cln @ \
                     (np.eye(n_state) - Kk_cln @ Hk_cln).T + Kk_cln @ R_noise @ Kk_cln.T
        P_cln2     = 0.5 * (P_cln2 + P_cln2.T) + 1e-12 * np.eye(n_state)

        if t >= T_warmup:
            online_t = t - T_warmup
            res_ekf_att[online_t] = r_att
            res_ekf_cln[online_t] = r_cln

    print(f"    EKF done.")

    # --------------------------------------------------------
    # Step E: TimesFM residuals (buffer protection for attacked)
    # --------------------------------------------------------
    res_fm_att = np.zeros((T_online_trial, m_meas))
    res_fm_cln = np.zeros((T_online_trial, m_meas))
    y_buffer   = y_att.copy()

    for t in range(T_warmup, T_trial):
        online_t = t - T_warmup
        start    = max(0, t - context_len)

        # attacked — use protected buffer
        contexts_att = [y_buffer[start:t, ch] for ch in range(m_meas)]
        pf_att, _    = model.forecast(horizon=1, inputs=contexts_att)
        y_hat        = pf_att[:, 0]
        res_fm_att[online_t] = y_att[t] - y_hat

        chi2_now = res_fm_att[online_t] @ Sigma_fm_inv @ res_fm_att[online_t]
        if chi2_now > threshold and t >= attack_start:
            y_buffer[t] = buffer_weight * y_hat + (1 - buffer_weight) * y_att[t]
        else:
            y_buffer[t] = y_att[t]

        # clean
        contexts_cln = [y_cln[start:t, ch] for ch in range(m_meas)]
        pf_cln, _    = model.forecast(horizon=1, inputs=contexts_cln)
        res_fm_cln[online_t] = y_cln[t] - pf_cln[:, 0]

    print(f"    TimesFM done.")

    # --------------------------------------------------------
    # Step F: chi2 statistics
    # --------------------------------------------------------
    chi2_fm_att  = np.array([res_fm_att[t]  @ Sigma_fm_inv @ res_fm_att[t]
                              for t in range(T_online_trial)])
    chi2_fm_cln  = np.array([res_fm_cln[t]  @ Sigma_fm_inv @ res_fm_cln[t]
                              for t in range(T_online_trial)])
    chi2_ekf_att = np.array([res_ekf_att[t] @ np.linalg.solve(S_store_t[T_warmup + t], res_ekf_att[t])
                              for t in range(T_online_trial)])
    chi2_ekf_cln = np.array([res_ekf_cln[t] @ np.linalg.solve(S_store_t[T_warmup + t], res_ekf_cln[t])
                              for t in range(T_online_trial)])

    # store only attack window
    chi2_fm_att_trials[trial]  = chi2_fm_att[T_clean:]
    chi2_fm_cln_trials[trial]  = chi2_fm_cln[T_clean:]
    chi2_ekf_att_trials[trial] = chi2_ekf_att[T_clean:]
    chi2_ekf_cln_trials[trial] = chi2_ekf_cln[T_clean:]

    if trial == 0:
        y_plot_att = y_att.copy()
        y_plot_cln = y_cln.copy()

    print(f"  Trial {trial+1}/{N_trials} done")

print("\nDone.")

# FAR and DR
FAR_fm  = np.mean(chi2_fm_cln_trials  > threshold)
FAR_ekf = np.mean(chi2_ekf_cln_trials > threshold)
DR_fm   = np.mean(np.any(chi2_fm_att_trials  > threshold, axis=1))
DR_ekf  = np.mean(np.any(chi2_ekf_att_trials > threshold, axis=1))
print(f"TimesFM — FAR: {FAR_fm:.3f} | DR: {DR_fm:.3f}")
print(f"EKF     — FAR: {FAR_ekf:.3f} | DR: {DR_ekf:.3f}")

# ============================================================
# PLOT
# ============================================================
fig = plt.figure(figsize=(14, 10.5))
gs  = gridspec.GridSpec(3, 2, figure=fig)

ax0        = fig.add_subplot(gs[0, :])
omega2_idx = 1
t_full     = np.arange(T_trial)
ax0.plot(t_full, y_plot_att[:, omega2_idx], '-o', color='steelblue',
         lw=0.8, markersize=2, label='Bus 2 freq (attacked)')
ax0.axvspan(T_warmup, T_warmup + T_clean-1,
            alpha=0.12, color='green', label='Recording window')

ax0.axvline(T_warmup, color='green', linestyle='--', lw=1.2)
ax0.axvline(T_warmup + T_clean-1, color='green', linestyle='--', lw=1.2)

ax0.plot(t_full, y_plot_cln[:, omega2_idx], '+-', color='gray',
         lw=0.8, alpha=0.6, label='Bus 2 freq (clean)')
ax0.axvline(attack_start, color='red', linestyle='--', lw=1.5, label='Replay start')
ax0.axvline(attack_start + attack_duration-1, color='darkred',
            linestyle='--', lw=1.5)
ax0.axvspan(attack_start, attack_start + attack_duration, alpha=0.08, color='red')
ax0.set_xlabel('Time step')
ax0.set_ylabel('$\\omega_2$ measurement')
ax0.set_title('IEEE 14-bus — replay attack on all sensors')
ax0.legend(fontsize=8)
ax0.grid(alpha=0.3)

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

ax_fm_cln = fig.add_subplot(gs[1, 0])
ax_fm_att = fig.add_subplot(gs[1, 1])
plot_band(ax_fm_cln, chi2_fm_cln_trials, color='steelblue',
          title='TimesFM — no attack', t_offset=attack_start)
plot_band(ax_fm_att, chi2_fm_att_trials, color='steelblue',
          title=f'TimesFM — replay attack (duration={attack_duration})',
          t_offset=attack_start)
ax_fm_cln.set_xlabel('Time step')
ax_fm_att.set_xlabel('Time step in attack window')

ax_ekf_cln = fig.add_subplot(gs[2, 0])
ax_ekf_att = fig.add_subplot(gs[2, 1])
plot_band(ax_ekf_cln, chi2_ekf_cln_trials, color='darkorange',
          title='EKF — no attack', t_offset=attack_start)
plot_band(ax_ekf_att, chi2_ekf_att_trials, color='darkorange',
          title=f'EKF — replay attack (duration={attack_duration})',
          t_offset=attack_start)
ax_ekf_cln.set_xlabel('Time step')
ax_ekf_att.set_xlabel('Time step in attack window')

y_max_cln = max(ax_fm_cln.get_ylim()[1], ax_ekf_cln.get_ylim()[1])
y_max_att = max(ax_fm_att.get_ylim()[1], ax_ekf_att.get_ylim()[1])
ax_fm_cln.set_ylim(0, y_max_cln);  ax_ekf_cln.set_ylim(0, y_max_cln)
ax_fm_att.set_ylim(0, y_max_att);  ax_ekf_att.set_ylim(0, y_max_att)

plt.suptitle(
    f'IEEE 14-bus Replay Attack | Number of trials={N_trials} | Context length={context_len} | FAR ($\\alpha$)={alpha}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('ieee14_replay_attack.pdf', bbox_inches='tight')
plt.show()
