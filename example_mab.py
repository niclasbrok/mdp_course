import numpy as np
import matplotlib
try:
    matplotlib.use('PyQt4')
except:
    pass
import matplotlib.pyplot as plt

np.random.seed(1)

lw = 4.0
fs = 15
matplotlib.rc('xtick', labelsize=fs)
matplotlib.rc('ytick', labelsize=fs)
matplotlib.rc('axes', titlesize=fs + 15)
matplotlib.rc('axes', labelsize=fs + 5)
matplotlib.rc('legend', fontsize=fs)

K = 50
B = 100
N = 10 * K
r = np.random.beta(2, 5, (20 * N, K, 1))
r[:, -1] = np.random.beta(7, 5, (20 * N, 1))
r = np.squeeze(r)
c = np.random.beta(2, 2, (N, K, 1))
c = np.squeeze(c)

fig = plt.figure(figsize=(18, 7))
ax = fig.add_subplot(111)
ax.set_title('Cumulative Distributions of Rewards')
ax.set_xlabel('Reward')
ax.set_ylabel('Freq.')
ax.hist(r[:, 0], bins=250, density=True, cumulative=True, label='Sh*t Distribution')
ax.hist(r[:, -1], bins=250, density=True, cumulative=True, label='Best Distribution')
ax.grid()
ax.legend()
fig.show()

t = np.arange(1, N + 1)
i_ucb = np.zeros(N)
i_ucb_r = np.zeros((N, K))
i_ucb_r[:] = np.nan
i_ucb_c = np.zeros((N, K))
i_ucb_c[:] = np.nan
i_random = np.random.randint(0, K, N)
i_random_r = np.zeros(N)
i_random_c = np.zeros(N)

# UCB algorithm
# Run through all machines to get data
for k in range(0, K):
    i_ucb[k] = k
    i_ucb_r[k, k] = r[k, k]
    i_ucb_c[k, k] = c[k, k]

for k in range(K, N):
    machine_reward = np.nanmean(i_ucb_r[0:k, :], 0)
    machine_cost = np.nanmean(i_ucb_c[0:k, :], 0)
    machine_ratio = np.divide(machine_reward, machine_cost)
    machine = np.argmax(machine_ratio)
    if machine.size == 1:
        i_ucb[k] = machine
    else:
        i_ucb[k] = machine[0]
    i_ucb_r[k, int(i_ucb[k])] = r[k, int(i_ucb[k])]
    i_ucb_c[k, int(i_ucb[k])] = c[k, int(i_ucb[k])]

# Random guessing algorithm
for k in range(0, N):
    i_random_r[k] = r[k, i_random[k]]
    i_random_c[k] = c[k, i_random[k]]

fig = plt.figure(figsize=(18, 7))

ax = fig.add_subplot(111)

ax.set_title('Performance of UCB Algorithm (K={0:d})'.format(K))
ax.set_xlabel('Time, t')
ax.set_ylabel('Accumulated Wealth')

ax.plot(t, B + np.cumsum(np.nansum(i_ucb_r, 1)) - np.cumsum(np.nansum(i_ucb_c, 1)), label='UCB Algorithm', linewidth=lw)
ax.plot(t, B + np.cumsum(i_random_r) - np.cumsum(i_random_c), label='Random Guessing', linewidth=lw)
ax.grid()
ax.set_ylim(bottom=0)
ax.legend(ncol=2, loc='lower left')

fig.show()
