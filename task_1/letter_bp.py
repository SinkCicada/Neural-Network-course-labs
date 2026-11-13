"""三层 BP 点阵字母识别实验。

运行：python letter_bp.py
NumPy 用于矩阵运算；BP 求导、梯度下降、阈值识别均自行实现。
"""

import csv
from itertools import combinations
from pathlib import Path

import numpy as np


# 一、实验参数。每次比较只改变隐含层节点数。
HIDDEN_SIZES = (3, 4, 5, 6, 7)
EPOCHS = 30000
LEARNING_RATE = 0.3
SEED = 7
ON = 0.9
OFF = 0.1
TESTS_PER_LETTER = 20  # 每个字母、每种扰动程度的不同测试模板数
DEMO_HIDDEN = 5        # 预先指定用于展示输出的网络，不根据测试成绩挑选
OUTPUT_DIR = Path(__file__).resolve().parent / "letter_bp_results"

# 自制标准模板：7 行 × 5 列，1 为黑色、0 为白色，直接使用这些值。
PATTERNS = {
    "A": ["00100", "01010", "10001", "10001", "11111", "10001", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
}
LETTERS = tuple(PATTERNS)


def to_vector(rows):
    """按列扫描：左上角向下，再进入下一列。"""
    matrix = np.array([[int(bit) for bit in row] for row in rows], dtype=float)
    return matrix.T.reshape(1, 35)  # 只排列像素，不改变像素数值


def to_image(vector):
    """把列扫描向量还原为点阵，供测试展示。"""
    return np.asarray(vector).reshape(5, 7).T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def recognize(output, on=ON, off=OFF):
    """唯一输出 >on 且其余全部 <off 才接受；默认仍为 0.9/0.1。"""
    if not 0 <= off < on <= 1:
        raise ValueError("阈值必须满足 0 <= off < on <= 1")
    output = np.asarray(output).reshape(-1)
    active = np.flatnonzero(output > on)
    if len(active) == 1:
        index = int(active[0])
        if np.all(np.delete(output, index) < off):
            return LETTERS[index]
    return "?"  # 阈值不满足，无法确定；不强制取最大值


class myBP:
    """沿用原代码的 w、b、v、c 和 forward/backward/train 结构。"""

    def __init__(self, x_num, y_num, h_num, seed=SEED):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-1, 1, (x_num, h_num))
        self.b = rng.uniform(-1, 1, (1, h_num))
        self.v = rng.uniform(-1, 1, (h_num, y_num))
        self.c = rng.uniform(-1, 1, (1, y_num))

    def forward(self, x):
        # 常数节点 x0=1：它的连接权重等价于单独存放的偏置 b、c。
        h = sigmoid(x.dot(self.w) + self.b)
        l = sigmoid(h.dot(self.v) + self.c)
        return l, h

    def backward(self, x, y):
        l, h = self.forward(x)
        # 平方误差 E=1/2*(t-y_hat)^2。本程序令 delta_l=y_hat-t。
        # 若用 e=t-y_hat，则写正向增量；这里减去梯度，两者等价。
        delta_l = l - y
        delta_c = delta_l * l * (1 - l)       # 输出层 dE/dz
        delta_v = h.T.dot(delta_c)
        # 按链式法则求导：对每个隐含节点，累加全部后续连接的误差贡献。
        # 对单个样本：delta_h[j] = Σ_k delta_c[k] * v[j,k]。
        delta_h = delta_c.dot(self.v.T)
        delta_b = delta_h * h * (1 - h)      # 隐含层 dE/dz
        delta_w = x.T.dot(delta_b)

        # 将原代码逐样本累加求平均，等价地写成批量矩阵运算。
        n = len(x)
        return (delta_c.sum(axis=0, keepdims=True) / n,
                delta_v / n,
                delta_b.sum(axis=0, keepdims=True) / n,
                delta_w / n)

    def train(self, train_x, train_y, times, r):
        history = []
        for epoch in range(times + 1):
            if epoch == 0 or epoch == 1 or epoch % 100 == 0 or epoch == times:
                output, _ = self.forward(train_x)
                # E = (1/2N) Σ样本 Σ输出 (预测值-目标值)^2，与上述梯度一致。
                error = float(0.5 * np.sum((output - train_y) ** 2) / len(train_x))
                accepted = sum(recognize(row) == letter
                               for row, letter in zip(output, LETTERS))
                history.append([epoch, error, accepted])
                if epoch <= 1 or epoch % 5000 == 0 or epoch == times:
                    print(f"H={self.w.shape[1]} 轮次={epoch:5d} "
                          f"E={error:.6f} 标准模板识别={accepted}/6")
            if epoch == times:
                break

            dc, dv, db, dw = self.backward(train_x, train_y)
            # 所有梯度计算完之后，统一更新；反传使用更新前的 v。
            self.c -= r * dc
            self.v -= r * dv
            self.b -= r * db
            self.w -= r * dw
        return history


def make_tests(train_x):
    """干净模板以及未参与训练的扰动样本；每次精确翻转 k 个不同像素。"""
    rng = np.random.default_rng(SEED + 1)
    tests = []
    for k in (0, 1, 2, 3):
        options = list(combinations(range(35), k))
        for index, letter in enumerate(LETTERS):
            count = 1 if k == 0 else TESTS_PER_LETTER
            selected = rng.choice(len(options), size=count, replace=False)
            for sample_id, option_index in enumerate(selected):
                vector = train_x[index].copy()
                pixels = list(options[int(option_index)])
                vector[pixels] = 1 - vector[pixels]
                tests.append((k, letter, sample_id, vector))
    return tests


def test_model(model, tests, hidden_size):
    summary = []
    details = []
    for k in (0, 1, 2, 3):
        correct = wrong = rejected = 0
        for flips, actual, sample_id, vector in tests:
            if flips != k:
                continue
            scores = model.forward(vector.reshape(1, 35))[0][0]
            predicted = recognize(scores)
            if predicted == actual:
                correct += 1
            elif predicted == "?":
                rejected += 1
            else:
                wrong += 1
            bits = "".join(str(int(bit)) for bit in vector)
            details.append([hidden_size, flips, actual, sample_id, predicted,
                            bits, *scores.tolist()])
        total = correct + wrong + rejected
        summary.append([hidden_size, k, total, correct, wrong, rejected,
                        correct / total, wrong / total, rejected / total])
        print(f"  翻转{k}像素：正确={correct}/{total} "
              f"错误={wrong}/{total} 无法确定={rejected}/{total}")
    return summary, details


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def draw_figures(histories, summary, tests, demo_model):
    # 绘图库只画图，不参与神经网络计算。
    import matplotlib
    matplotlib.use("Agg")  # 保存图片，不阻塞程序或打开窗口
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for hidden_size, history in histories.items():
        values = np.array(history)
        ax.semilogy(values[:, 0], values[:, 1], label=f"H = {hidden_size}")
    ax.set(xlabel="Training epoch", ylabel="Mean sample squared error E",
           title="Handwritten BP: 35 inputs - H hidden - 6 outputs")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "training_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for hidden_size in histories:
        rows = np.array([row for row in summary if row[0] == hidden_size])
        axes[0].plot(rows[:, 1], rows[:, 6], marker="o", label=f"H = {hidden_size}")
        axes[1].plot(rows[:, 1], rows[:, 8], marker="o", label=f"H = {hidden_size}")
    for ax, title in zip(axes, ("Correct recognition rate", "Rejection rate")):
        ax.set(xlabel="Number of flipped pixels", ylabel="Rate", title=title,
               xticks=[0, 1, 2, 3], ylim=(-0.03, 1.03))
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.savefig(OUTPUT_DIR / "noise_test.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(6, 4, figsize=(8, 11), constrained_layout=True)
    for row, letter in enumerate(LETTERS):
        for k in range(4):
            vector = next(v for flips, actual, sample_id, v in tests
                          if flips == k and actual == letter and sample_id == 0)
            scores = demo_model.forward(vector.reshape(1, 35))[0][0]
            predicted = recognize(scores)
            ax = axes[row, k]
            ax.imshow(to_image(vector), cmap="gray_r", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
            ax.grid(which="minor", color="#999999", linewidth=0.4)
            ax.tick_params(which="both", bottom=False, left=False,
                           labelbottom=False, labelleft=False)
            ax.set_title(f"{letter}, flips={k} -> {predicted}", fontsize=10,
                         color="#156c42" if predicted == letter else "#ad4027")
    fig.suptitle(f"Test examples with H={DEMO_HIDDEN}; ? = threshold not met")
    fig.savefig(OUTPUT_DIR / "test_examples.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    train_x = np.concatenate([to_vector(PATTERNS[letter]) for letter in LETTERS])
    train_y = np.eye(len(LETTERS))
    tests = make_tests(train_x)  # 只生成一次，各网络使用同一测试集
    histories, summary, details, models = {}, [], [], {}
    print("35-H-6 三层 BP，输出顺序：", LETTERS)
    print(f"学习率={LEARNING_RATE}，训练轮数={EPOCHS}；"
          f"恰好一个输出>{ON}且其余<{OFF}才接受识别。")
    for hidden_size in HIDDEN_SIZES:
        print(f"\n=== 隐含层节点数 H={hidden_size} ===")
        model = myBP(35, 6, hidden_size)
        histories[hidden_size] = model.train(train_x, train_y, EPOCHS, LEARNING_RATE)
        rows, predictions = test_model(model, tests, hidden_size)
        summary.extend(rows)
        details.extend(predictions)
        models[hidden_size] = model

    print(f"\nH={DEMO_HIDDEN} 标准模板的六个输出值：")
    for letter, vector in zip(LETTERS, train_x):
        scores = models[DEMO_HIDDEN].forward(vector.reshape(1, 35))[0][0]
        print(f"{letter}: " + " ".join(f"{score:.4f}" for score in scores)
              + f" -> {recognize(scores)}")

    history_rows = [[h, *row] for h, history in histories.items() for row in history]
    write_csv(OUTPUT_DIR / "training_history.csv",
              ["hidden", "epoch", "mean_sample_error", "clean_correct"], history_rows)
    write_csv(OUTPUT_DIR / "test_summary.csv",
              ["hidden", "flipped_pixels", "total", "correct", "wrong", "rejected",
               "correct_rate", "wrong_rate", "rejection_rate"], summary)
    write_csv(OUTPUT_DIR / "test_predictions.csv",
              ["hidden", "flipped_pixels", "actual", "sample_id", "predicted",
               "column_scan_bits", *LETTERS], details)
    draw_figures(histories, summary, tests, models[DEMO_HIDDEN])
    print(f"\n误差、预测记录和三张实验图已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
