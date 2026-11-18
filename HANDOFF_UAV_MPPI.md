# UAV 编队对照交付说明

更新日期：2026-11-18。原 `HANDOFF.md` 是另一项 Tube-RRT 工作的交付，请保留。

## 当前目标与约束

- 三路线比较：论文统一缩放前端 + 分布式 MPPI、论文仿射前端 + 分布式 MPPI、局部仿射 RRT* 前端 + **同一套论文 MPPI 后端**。第三路线旧集中式 Bézier 后端仍可选，不得删除。
- 用户要求保留原始队形到达条件（A=I），关注实际到达、失败原因和安全；不能把部分轨迹安全或成功靠近说成完成。不要盲目调参、改造论文 MPPI，或把失败结果覆盖。需要核验证据再做后续修复。
- 已诊断并修复 RRT + MPPI seed 1 的 `no_safe_nominal_root` 后停滞：这是**前端名义根无法安全接入**，不是 MPPI 求解失败。证据和闭环复验见下文。

## 模型与实验口径

- 本轮 UAV 是论文 Eq.4 的十状态 `[p3,v3,a3,yaw]`、控制 `[jerk3,yaw_rate]`，固定高度 1 m 平面场景对照；并未证明作者实验源码就是二维。MPPI 执行按 0.02 s 前向 Euler；每个 0.2 s 预测步等效组合 10 个恒控制 Euler 子步。这是 Eq.23 的数值解释，不冒称作者源码。
- 执行 50 Hz、感知 5 Hz、前端 1 Hz、后端优化 1 Hz；MPPI 预测步长 0.2 s，复用控制序列中的中间 0.2 s 命令。速度阈值 1.5 m/s 来自论文；加速度阈值 1 m/s²、噪声等为本地假设，MPPI 对速度/加速度使用软代价，不做硬裁剪。
- 相同地图：50×40 m、50 个轴对齐方柱、宽 0.3 m、map seed 1；三角 6 机最近邻 3.5 m，起终质心 (5,20)→(45,20)。柱宽和间距是对照假设，非论文明确数值。
- 第三路 RRT 引导参考速 0.55 m/s 为本地假设；逐边插值且保留拐点。采样、SFC 和实际接入检查不等于后端动态硬安全证明。
- 论文 MPPI 用障碍与互碰**软代价**，不是 SFC 硬约束优化。SFC 是此仓库前端生成及连接认证的一部分；前端如何对应论文的 SFC 语义此前存在争议，尚未为此擅改。

## 代码位置

- `formation/paper_formation/simulation.py`：共享仿真调度，`frontend='paper'|'rrt'`；`uav_mppi.py`、`uav_dynamics.py`：分布式 MPPI 与 UAV 状态更新；`guidance.py`：论文前端。
- `formation/local_affine/mppi_frontend.py`：RRT→MPPI 适配；`rrt.py`、`geometry.py`：局部树与边认证；`trajectory*.py`、`goal_time.py`、`local_affine/simulation.py`：可选 Bézier 后端。
- RRT 先拟合实际状态作为规划根；拟合形状不合法或无法安全接入时，使用上次成功首参考的 A，按当前真实机器人位置锚定名义构型，再做实际状态到名义根的逐车单 SFC 和连续车间距检查。失败记录 `no_safe_nominal_root`；不修改实际状态或夹紧物理轨迹。`planning.json` 保存 `root_source`、`actual_fit_root` 和树/引导证据。
- 保留的 Bézier 诊断：`geometry.certify_edge` 用 `argmax` 任意选第一个公共 SFC；固定失败快照中改选余量大的公共区域曾恢复部分可行。时长分配和控制点凸包也有保守性。**这些区域选择修复未上线**；不应将其当成当前 RRT + MPPI 的修复。

## 当前正式六组结果

`results/three_routes/uav_mppi_formal_20261118_150646/` 是当前结果；三路线均使用 MPPI。论文 uniform 前端的传播半径为 `r=0.3(1+γ)`，论文 affine 前端仍为 `r=0.1(1+γ)`，其中 γ 均匀采样于 `[0,1)`。六组均使用 `gcopter_old_firi_planar_disk_seed_strict_separation_v2`：原生 FIRI 对已分离障碍点采用严格正侧判定，原生库已重建。每组的 `configuration.json` 记录实际参数和区域版本。

| 路线 | seed | 终止 | 仿真时间 | 质心距目标 | 最大单机终点误差 |
|---|---:|---|---:|---:|---:|
| 论文 uniform + MPPI | 0 | goal_reached | 243.8 s | 0.2183 m | 0.3961 m |
| 论文 uniform + MPPI | 1 | goal_reached | 213.0 s | 0.2096 m | 0.3967 m |
| 论文 affine + MPPI | 0 | goal_reached | 265.0 s | 0.2407 m | 0.3906 m |
| 论文 affine + MPPI | 1 | goal_reached | 418.4 s | 0.1786 m | 0.3938 m |
| RRT affine + MPPI | 0 | goal_reached | 156.2 s | 0.0389 m | 0.1238 m |
| RRT affine + MPPI | 1 | goal_reached | 161.4 s | 0.0185 m | 0.1130 m |

六组均满足最大单机误差 < 0.4 m、最大单机速度 < 0.1 m/s 的终点条件。各组 `audit.json` 的连续安全、Euler 动力学、速度和加速度检查通过，区域锚点、引导边及实际状态接入均无未认证项；连续安全指相邻离散 UAV 状态位置之间的直线插值。两组 uniform 及两组 affine 终点最大误差均接近 0.4 m，应保留这一余量信息。完整数值见正式目录的 `summary.csv`；`comparison.png` 和 `seed{0,1}_comparison.mp4` 为本轮可视化。

uniform 的 0.3 m 先在旧版 FIRI 下按固定 19 s 快照和两组完整闭环检验，结果保留于 `results/three_routes/uniform_rmin_0p3/`；该临时对照不是上述新版 FIRI 六组。固定快照中旧路径失效与首机锚定的名义中心前移、第五架无法接入当前 SFC 有关，不能仅凭这一快照将所有失败归因于同一原因。

## 原始对照结果（保留失败证据）

目录 `results/three_routes/uav_rrt_mppi/`，各行 seed 指规划随机种子；距离是终止时质心到目标距离。全部已执行轨迹段的独立连续安全、UAV 动力学和前端引导区域审计通过；这是截断轨迹的结论，不代表所有路线到达。

| 路线 | seed | 终止 | 仿真时间 | 质心距目标 |
|---|---:|---|---:|---:|
| 论文 uniform + MPPI | 0 | stagnation | 51.0 s | 34.493 m |
| 论文 uniform + MPPI | 1 | sfc_failed | 90.0 s | 12.575 m |
| 论文 affine + MPPI | 0 | goal_reached | 265.0 s | 0.241 m |
| 论文 affine + MPPI | 1 | sfc_failed | 21.0 s | 34.292 m |
| RRT affine + MPPI | 0 | goal_reached | 156.2 s | 0.039 m |
| RRT affine + MPPI | 1 | stagnation | 56.0 s | 35.084 m |

- 四个论文前端基线与 `uav_comparison` 对应 states/controls 逐元素相同，证据 `uav_rrt_mppi/baseline_consistency.json`。`uav_comparison` 的第三路线是 **RRT + 集中式 Bézier**，不可误认成当前 RRT + MPPI；`uav_rrt_mppi_initial_adapter/` 是名义根恢复前的第三路线旧结果。
- `uav_rrt_mppi/summary.csv`、`paper_metrics.csv/json`、`paper_metrics_timeseries.png`、`seed{0,1}_comparison.mp4` 是该历史批次的汇总、论文口径指标与视频。论文指标中的 `t_sol` 是本地全队后端每轮时间代理，不可与论文分布式 GPU 时间直接等同；失败的 `t_fly` 为空。

## 名义根恢复后的闭环复验

旧 seed 1 在 56 s 停滞的直接原因是实际状态无法安全接入当时选择的名义根：原回退使用上一参考形状并固定首车位置；旧结果中的 32 次根接入失败快照均可用同一形状、按全队位置计算的质心平移安全接入。这是前端根选择的问题，不是 MPPI 求解失败。最终实现保留首车锚定作为首选，在其接入失败时尝试最小化全队平方接入距离的质心平移，仍须通过逐车单 SFC 和连续车间距检查。

最终代码的结果保存在 `results/three_routes/uav_rrt_centroid_final/`，旧失败结果保留在 `uav_rrt_mppi/`：

| RRT + MPPI | 终止 | 仿真时间 | 质心距目标 |
|---|---|---:|---:|
| seed 0 | goal_reached | 156.2 s | 0.03894 m |
| seed 1 | goal_reached | 161.4 s | 0.01854 m |

seed 0 与旧 `uav_rrt_mppi`、seed 1 与 `uav_rrt_centroid_recovery_v1` 的轨迹 NPZ 数值字段逐元素一致。seed 1 在 24、25、26 s 使用质心恢复；其独立审计检查 972 个区域、1916 条引导边，实际接入和引导边均通过，Euler 动力学残差为零，速度与加速度满足阈值。按相邻离散位置直线插值，障碍净距下界 0.5003 m、车间净距下界 2.5717 m；审计记录在 `uav_rrt_centroid_recovery_v1/rrt_affine/seed_1/audit.json`。

## 可复现入口

环境 `/home/eai/projects/env-rebuilt`；本轮无需安装新包。单组实验（勿覆盖已有结果）：

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-formation \
/home/eai/projects/env-rebuilt/bin/python scripts/experiments/compare_formation_routes.py \
  --route rrt_affine --rrt-backend mppi --dynamics uav --seed 0 \
  --max-time 500 --output results/three_routes/NEW_RUN --progress
```

`--route paper_uniform|paper_affine|rrt_affine`；旧方案用 `--route rrt_affine --rrt-backend bezier`。完整六组生成后，每组独立审计，再汇总、算指标、渲染：

```bash
/home/eai/projects/env-rebuilt/bin/python scripts/paper_formation/audit_paper_formation.py \
  results/three_routes/NEW_RUN/rrt_affine/seed_0/trajectory.npz --regions \
  --output results/three_routes/NEW_RUN/rrt_affine/seed_0/audit.json
/home/eai/projects/env-rebuilt/bin/python scripts/diagnostics/summarize_formation_routes.py results/three_routes/NEW_RUN
MPLCONFIGDIR=/tmp/mpl-formation /home/eai/projects/env-rebuilt/bin/python scripts/diagnostics/paper_metrics.py --root results/three_routes/NEW_RUN
/home/eai/projects/env-rebuilt/bin/python scripts/visualization/render_three_routes.py --root results/three_routes/NEW_RUN
```

原生库构建入口：`formation/paper_formation/native/build_firi.sh` 和 `build_affine_sampler.sh`。仓库当前有未提交改动和新文件；用户自行处理 Git，接手时不要重置或覆盖他人工作。避免新增长文档和无依据的参数扫描。
