"""固定手写 BP 网络，研究输出阈值对正确、误识、拒识的影响。

运行：python threshold_experiment.py
只训练一个预先指定的 H=5 网络，所有阈值共用权重和输出。
在验证集上优先选择零误识、正确数最多的规则，测试集不参与选择。
结果写入 threshold_results。
"""
import csv
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import letter_bp as bp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "threshold_results"
HIDDEN = 5
SPLIT_SEED = 20261110
ON_VALUES = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
OFF_VALUES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
BASELINE = (0.90, 0.10)
# 单像素翻转每类只有35种，排除原测试的20种后只剩15种。
SPLIT_COUNTS = {1: (7, 8), 2: (20, 20), 3: (20, 20)}


def make_splits(train_x):
    """排除原有测试集、干净模板及跨集合重复点阵。"""
    used = {tuple(x) for x in train_x}
    used.update(tuple(v) for _, _, _, v in bp.make_tests(train_x))
    rng = np.random.default_rng(SPLIT_SEED)
    records = []
    for k, (n_val, n_test) in SPLIT_COUNTS.items():
        choices = list(combinations(range(35), k))
        for index, letter in enumerate(bp.LETTERS):
            picked = []
            for option in rng.permutation(len(choices)):
                vector = train_x[index].copy()
                positions = list(choices[int(option)])
                vector[positions] = 1 - vector[positions]
                key = tuple(vector)
                if key in used:
                    continue
                used.add(key)
                picked.append(vector)
                if len(picked) == n_val + n_test:
                    break
            if len(picked) != n_val + n_test:
                raise ValueError(f"{letter} 翻转{k}像素的剩余不同点阵不足")
            for i, vector in enumerate(picked):
                split = "validation" if i < n_val else "test"
                sample_id = i if split == "validation" else i - n_val
                records.append((split, k, letter, sample_id, vector))
    return records


def evaluate(records, scores, on, off):
    predictions = [bp.recognize(row, on, off) for row in scores]
    rows = []
    for k in (1, 2, 3, "all"):
        indices = [i for i, row in enumerate(records) if k == "all" or row[1] == k]
        correct = sum(predictions[i] == records[i][2] for i in indices)
        rejected = sum(predictions[i] == "?" for i in indices)
        total = len(indices)
        wrong = total - correct - rejected
        rows.append(dict(on=on, off=off, flipped_pixels=k, total=total,
                         correct=correct, wrong=wrong, rejected=rejected,
                         correct_rate=correct / total, wrong_rate=wrong / total,
                         rejection_rate=rejected / total))
    return rows, predictions


def save_rows(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_figures(validation_rows, comparison, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 11,
                         "axes.unicode_minus": False})
    all_rows = [row for row in validation_rows if row["flipped_pixels"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, key, title in zip(axes, ("correct_rate", "wrong_rate"),
                              ("验证集正确识别率 / %", "验证集误识率 / %")):
        values = np.array([[next(r[key] for r in all_rows if r["on"] == on and r["off"] == off)
                            for off in OFF_VALUES] for on in ON_VALUES]) * 100
        image = ax.imshow(values, cmap="Blues" if key == "correct_rate" else "Oranges",
                          vmin=0, vmax=100 if key == "correct_rate" else max(1, values.max()))
        for i, on in enumerate(ON_VALUES):
            for j, off in enumerate(OFF_VALUES):
                ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center",
                        color="white" if values[i,j] > 0.6 * image.norm.vmax else "black", fontsize=9)
        for pair, color, marker in ((BASELINE, "red", "s"), (selected, "green", "o")):
            ax.scatter(OFF_VALUES.index(pair[1]), ON_VALUES.index(pair[0]),
                       s=440, facecolors="none", edgecolors=color, marker=marker, linewidths=2)
        ax.set(xticks=range(len(OFF_VALUES)), xticklabels=OFF_VALUES,
               yticks=range(len(ON_VALUES)), yticklabels=ON_VALUES,
               xlabel="其他节点上限 off", ylabel="目标节点下限 on", title=title)
    fig.suptitle("H=5 固定权重：红框为基准，绿圈为验证集选定规则")
    fig.savefig(OUT / "validation_threshold_grid.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    rows = [row for row in comparison if row["flipped_pixels"] != "all"]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for key, label, color in (("correct_rate", "正确", "#45876b"),
                               ("wrong_rate", "误识", "#c65c54"),
                               ("rejection_rate", "拒识", "#c3c8ce")):
        values = np.array([r[key] for r in rows]) * 100
        ax.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    labels = [f"{'基准' if r['rule']=='baseline' else '选定'}：翻转{r['flipped_pixels']}像素" for r in rows]
    ax.set(xticks=x, xticklabels=labels, ylabel="样本占比 / %", ylim=(0, 107),
           title=f"留出测试集：基准0.9/0.1 与选定{selected[0]:g}/{selected[1]:g}")
    ax.legend(loc="upper center", ncol=3)
    fig.savefig(OUT / "heldout_threshold_comparison.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    train_x = np.concatenate([bp.to_vector(bp.PATTERNS[c]) for c in bp.LETTERS])
    train_y = np.eye(len(bp.LETTERS))
    records = make_splits(train_x)
    model = bp.myBP(35, 6, HIDDEN, seed=bp.SEED)
    history = model.train(train_x, train_y, bp.EPOCHS, bp.LEARNING_RATE)
    scores = model.forward(np.stack([r[4] for r in records]))[0]
    # 之后只缓存输出、改变判别规则；不再更新任何参数。
    validation_indices = [i for i, r in enumerate(records) if r[0] == "validation"]
    test_indices = [i for i, r in enumerate(records) if r[0] == "test"]
    validation = [records[i] for i in validation_indices]
    test = [records[i] for i in test_indices]
    validation_scores = scores[validation_indices]
    test_scores = scores[test_indices]
    grid = []
    for on, off in product(ON_VALUES, OFF_VALUES):
        grid.extend(evaluate(validation, validation_scores, on, off)[0])
    totals = [row for row in grid if row["flipped_pixels"] == "all"]
    # 预先规定：先最小化误识数，再最大化正确数；同分优先更严格的约束。
    best = max(totals, key=lambda r: (-r["wrong"], r["correct"], r["on"], -r["off"]))
    selected = (best["on"], best["off"])
    comparison, details, changes = [], [], []
    predictions_by_rule = {}
    for rule, pair in (("baseline", BASELINE), ("selected", selected)):
        rows, predictions = evaluate(test, test_scores, *pair)
        comparison.extend(dict(rule=rule, **row) for row in rows)
        predictions_by_rule[rule] = predictions
        for record, score, predicted in zip(test, test_scores, predictions):
            _, k, actual, sample_id, vector = record
            details.append(dict(rule=rule, on=pair[0], off=pair[1], flipped_pixels=k,
                                actual=actual, sample_id=sample_id, predicted=predicted,
                                column_scan_bits="".join(str(int(v)) for v in vector),
                                **dict(zip(bp.LETTERS, score.tolist()))))
    for i, record in enumerate(test):
        old, new = (predictions_by_rule[r][i] for r in ("baseline", "selected"))
        if old != new:
            changes.append(dict(flipped_pixels=record[1], actual=record[2], sample_id=record[3],
                                baseline=old, selected=new, **dict(zip(bp.LETTERS, test_scores[i].tolist()))))

    save_rows("validation_grid.csv", grid)
    save_rows("heldout_comparison.csv", comparison)
    save_rows("heldout_predictions.csv", details)
    if changes:
        save_rows("changed_decisions.csv", changes)
    save_rows("training_history.csv", [dict(epoch=e, error=err, clean_correct=n) for e, err, n in history])
    save_rows("split_outputs.csv", [dict(split=r[0], flipped_pixels=r[1], actual=r[2], sample_id=r[3],
                                         column_scan_bits="".join(str(int(v)) for v in r[4]),
                                         **dict(zip(bp.LETTERS, score.tolist()))) for r, score in zip(records, scores)])
    np.savez(OUT / "fixed_model.npz", w=model.w, b=model.b, v=model.v, c=model.c)
    config = dict(hidden=HIDDEN, epochs=bp.EPOCHS, learning_rate=bp.LEARNING_RATE,
                  model_seed=bp.SEED, split_seed=SPLIT_SEED,
                  on_values=ON_VALUES, off_values=OFF_VALUES, baseline=BASELINE,
                  selected=selected, validation_count=len(validation), test_count=len(test),
                  split_counts_per_letter=SPLIT_COUNTS,
                  validation_selected=best,
                  selected_on_grid_boundary=(selected[0] in (min(ON_VALUES), max(ON_VALUES))
                                             or selected[1] in (min(OFF_VALUES), max(OFF_VALUES))),
                  selection="min wrong, max correct, max on, min off; validation only",
                  excludes="six training templates and all original bp.make_tests samples",
                  limitation="single initialization and synthetic pixel flips around six templates; "
                             "selected pair is best only in this candidate grid, not a global optimum")
    (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_figures(grid, comparison, selected)
    print(f"\n验证集 {len(validation)}，留出测试集 {len(test)}；选定阈值 on={selected[0]}, off={selected[1]}")
    print("选定规则仅对应当前候选网格；不据此认定所有输入都适合放宽阈值。")
    for row in comparison:
        if row["flipped_pixels"] == "all":
            print(f"{row['rule']}: 正确={row['correct']}/{row['total']}，误识={row['wrong']}，拒识={row['rejected']}")
    print("结果目录：", OUT)


if __name__ == "__main__":
    main()
