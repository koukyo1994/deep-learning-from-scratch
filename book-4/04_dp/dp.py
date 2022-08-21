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

cnt = 0  # 何回更新したかを記録
threshold = 0.0001  # 更新量がこの値を下回ったら停止する

while True:
    new_V["L1"] = (
        0.5 * (r[("L1", "L1")] + 0.9 * V["L1"]) +
        0.5 * (r[("L1", "L2")] + 0.9 * V["L2"])
    )
    new_V["L2"] = (
        0.5 * (r[("L2", "L1")] + 0.9 * V["L1"]) +
        0.5 * (r[("L2", "L2")] + 0.9 * V["L2"])
    )

    # 更新された量の最大値
    delta = abs(new_V["L1"] - V["L1"])
    delta = max(delta, abs(new_V["L2"] - V["L2"]))

    V = new_V.copy()
    cnt += 1
    if delta < threshold:
        print(V)
        print(cnt)
        break
