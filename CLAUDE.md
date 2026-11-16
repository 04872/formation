# Project notes

- Python / Conda 环境：`/home/eai/projects/env-rebuilt`（从仓库根目录用 `../env-rebuilt/bin/python` 运行，或 `conda activate /home/eai/projects/env-rebuilt`）。不要使用 `../env`，该目录不存在。

- `formation/`：`maps` 地图、`planning` 路径与引导、`geometry` 队形与安全区域、`rrt` 搜索、`control` 控制，`paper_formation` 为论文适配；公共类型和指标留在根目录。
- `scripts/`：`paper_formation` 论文实验与验收、`experiments` 对照实验、`diagnostics` 诊断、`visualization` 绘图、`setup` 环境。示例：`python scripts/paper_formation/run_paper_formation.py --help`。
