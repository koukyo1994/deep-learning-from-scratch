import numpy as np


x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

# 普通に期待値を求める
e = np.sum(x * pi)
print(f"E_pi[x]: {e:.3f}")

# モンテカルロ法で期待値を推定する
n = 100
samples = []
for _ in range(n):
    samples.append(np.random.choice(x, p=pi))  # piを使ってサンプリング

mean = np.mean(samples)
var = np.var(samples)
print(f"MC: {mean:.3f} (var: {var:.3f})")

# 他の確率分布を使って重点サンプリングにより期待値を求める
b = np.array([1 / 3, 1 / 3, 1 / 3])
n = 100
samples = []

for _ in range(n):
    idx = np.arange(len(b))
    i = np.random.choice(idx, p=b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(rho * s)

mean = np.mean(samples)
var = np.var(samples)
print(f"IS: {mean:.3f} (var: {var:.3f})")

# より近い分布を用いて重点サンプリングする
b = np.array([0.2, 0.2, 0.6])
n = 100
samples = []

for _ in range(n):
    idx = np.arange(len(b))
    i = np.random.choice(idx, p=b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(rho * s)

mean = np.mean(samples)
var = np.var(samples)
print(f"IS: {mean:.3f} (var: {var:.3f})")
