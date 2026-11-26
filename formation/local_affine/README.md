# 局部仿射编队规划

`rrt.py` 在局部感知地图和共享 FIRI 安全区域内搜索仿射构型，`trajectory.py` 集中优化共享的分段五次 Bézier 轨迹，`simulation.py` 按滚动时域执行前缀。普通局部末端速度和加速度自由，前端折点用作软跟踪参考；真正目标端要求精确到达并停止；重规划失败时继续执行上一条有效轨迹；非静止轨迹耗尽仍无新解则终止并记录失败。使用机器人位移距离、沿边形状和障碍风险、ChooseParent 与 Rewire；连续环境和车间安全使用共同凸区域与定向分离约束。仿射形状边界在每段采用局部凸内近似。

```bash
MPLCONFIGDIR=/tmp/mpl-formation OPENBLAS_NUM_THREADS=1 /home/eai/projects/env-rebuilt/bin/python scripts/experiments/run_local_affine.py --scene random_pillars --obstacle-count 50 --mode affine --iterations 250 --execution-dt .02 --perception-period .2 --replan-period 1 --max-time 500 --progress --output results/local_affine/run
```

`--mode uniform|affine` 选择统一缩放或一般仿射；地图与初终质心默认分别为 50×40 m 和 (5,20)→(45,20)，三角六车间距 3.5 m、方柱宽 0.3 m 是自选假设。`trajectory.npz` 保存执行状态、即时速度/角速度、连续曲线控制点、区域约束与感知记录；`plans.json` 保存各轮树、空间安全区域和各段执行区间。

执行、感知和规划在同一同步仿真循环按各自仿真时间周期调度；规划计算耗时不会推进仿真时间。曲线段时长按执行步长取整。旧参数 `--max-steps`、`--replan-steps` 显式使用时分别按执行步长换算并覆盖 `--max-time`、`--replan-period`。

本轮对比显式采用 1 秒重规划，非终点主轨迹的估计时域不少于一个重规划周期加一个执行步长。时长由路径与初始运动状态估计并做必要条件检查，并非完整的时间分配优化。不再附加制动备用段。

差速车状态通过共享平坦输出曲线及导数解析恢复，静止时可原地转向；这不是包含执行器和跟踪误差的闭环控制实验。初版存在多次停走，不构成实时性能结论。速度与加速度使用 Bézier 导数控制点约束，角速度通过顺序凸化修正并以全段多项式数值极值检查验收，修正使用自适应极值位置约束，直接数值装配至 Clarabel；具体轨迹仍需独立验收；有限预算局部搜索不保证到达。

`metrics.json` 的 `timings_seconds` 分开记录感知、SFC、前端、后端和执行计算耗时；`simulation_duration_seconds` 是机器人运动的仿真时长。前端缓存节点几何并按代价下界剪枝。后端通过 `trajectory_template.py` 按段数和各段实际约束行数缓存参数化凸问题，跨轮更新路径、SFC 和初始状态，并为每次求解设置选定的时长；缓存不会改变对应优化问题的目标与约束，不补齐约束行。当前主目标包含前端位置与参考速度的积分跟踪及加速度平滑。缓存归各次仿真独立所有，日志中的 `template_cache_hit` 记录复用情况。

三路线 UAV 对比入口：`scripts/experiments/compare_formation_routes.py --dynamics uav --route paper_uniform|paper_affine|rrt_affine --seed 0 --output results/three_routes/uav_comparison`。固定高度下采用论文位置/速度/加速度积分链与独立偏航，速度上限 1.5 m/s，加速度上限 1 m/s² 为实验假设；集中式解析输出 p/v/a/jerk，不施加差速曲率约束，局部参考速度仍为 0.55 m/s。MPPI 保留软动力学代价，集中式保留硬速度/加速度界。两者执行模型与数值步长详见结果配置。

第三条路线现默认使用 `compare_formation_routes.py --route rrt_affine --rrt-backend mppi`，由 `mppi_frontend.py` 将 RRT* 折线转成引导点，复用论文 MPPI 仿真循环；`--rrt-backend bezier` 保留上述集中式后端。新对比结果位于 `results/three_routes/uav_rrt_mppi/`。

历史可见性图选枝通过 `--history-decision` 开启，仅适用于 `--route rrt_affine --rrt-backend mppi`；默认关闭，保留原 RRT* 评分。以下分别运行非凸死胡同和随机柱场景；去掉 `--history-decision` 并使用另一个输出目录，可运行相同种子的原评分对照。

```bash
MPLCONFIGDIR=/tmp/mpl-formation OPENBLAS_NUM_THREADS=1 /home/eai/projects/env-rebuilt/bin/python scripts/experiments/compare_formation_routes.py --route rrt_affine --rrt-backend mppi --dynamics uav --scene history_deadend --seed 0 --history-decision --progress --output results/history_deadend
MPLCONFIGDIR=/tmp/mpl-formation OPENBLAS_NUM_THREADS=1 /home/eai/projects/env-rebuilt/bin/python scripts/experiments/compare_formation_routes.py --route rrt_affine --rrt-backend mppi --dynamics uav --scene random_pillars --map-seed 1 --seed 0 --history-decision --progress --output results/history_random_pillars
```

`planning.json` 中记录每个合格候选末端的树代价、逐成员欧氏距离、图上距离和评分，并记录原评分与实际选中末端。当任一成员的图查询缺失时，本轮所有候选统一使用原欧氏距离评分；否则使用历史图距离的成员均方根，权重与原评分相同。图查询允许未被已知障碍阻断的未知连接作为尝试性估计，它不认证后续编队运动；实际引导始终来自当前逐成员 SFC 认证的树。目标直连和没有合格候选的情况沿用原前端处理。

历史联合路径模式使用 `--history-joint-path`，与 `--history-decision` 互斥，仅适用于 RRT 仿射前端和 MPPI 后端。它保存合格引导上的已认证联合运动，以短连接接入当前工作图；将任一成员的感知圆盘覆盖到尚未观测栅格的配置作为观测边界，按图上到边界的运动代价加逐成员余程评价候选。目标配置的终端值为零；所有候选均无法到达边界或目标时整轮回退到原评分。只有实际运动获得的观测更新累计地图。`planning.json` 的 `history_decision` 记录边界数量、`round_fallback` 和每个候选的 `observation_boundary` 等诊断。

闭环消融可同时传入 `--history-joint-path --history-current-motion-only`：累计地图、历史 SFC 和逐成员启发式不变，评价图只使用当轮运动。配置会记录 `history_current_motion_only`。
