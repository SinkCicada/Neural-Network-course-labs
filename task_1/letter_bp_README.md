# 点阵字母识别

`letter_bp.py` 包含数据模板、三层 BP 网络、训练和绘图。输入是 7×5 的二值点阵，按列展开为 35 维向量；六个输出对应 A、E、F、H、L、T。

采用 Sigmoid 激活、平方误差和全批量梯度下降。隐含层节点分别取 3、4、5、6、7，学习率为 0.3，训练 30000 轮。六个干净模板用于训练，随机翻转 1～3 个像素生成测试样本。默认识别规则要求唯一输出大于 0.9，其余输出均小于 0.1，否则拒识。

`threshold_experiment.py` 固定 H=5 的网络，比较不同 ON/OFF 阈值。在验证集上优先选择零误识、正确数最多的规则，再使用留出测试集检验。配置及完整数据保存在 `threshold_results`。

在仓库根目录运行：

```powershell
python task_1/letter_bp.py
python task_1/threshold_experiment.py
```

训练曲线及隐含层对比保存在 `letter_bp_results`。脚本再次运行时会更新对应结果文件。
