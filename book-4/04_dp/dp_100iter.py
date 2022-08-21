# 推定量の初期化
V = {"L1": 0.0, "L2": 0.0}
new_V = V.copy()  # Vのコピー

# 割引率
gamma = 0.9

# 報酬関数
r = {
    # (現在の状態, 次の状態): 報酬 の形式になっている辞書
    ("L1", "L1"): -1.0,
    ("L1", "L2"):  1.0,
    ("L2", "L1"):  0.0,
    ("L2", "L2"): -1.0,
}

for _ in range(100):
    new_V["L1"] = (
        0.5 * (r[("L1", "L1")] + 0.9 * V["L1"]) +
        0.5 * (r[("L1", "L2")] + 0.9 * V["L2"])
    )
    new_V["L2"] = (
        0.5 * (r[("L2", "L1")] + 0.9 * V["L1"]) +
        0.5 * (r[("L2", "L2")] + 0.9 * V["L2"])
    )
    V = new_V.copy()
    print(V)
