# 局部仿射编队规划

`rrt.py` 在局部感知地图和共享 FIRI 安全区域内搜索仿射构型，`trajectory.py` 集中优化共享的分段五次 Bézier 轨迹，`simulation.py` 按滚动时域执行前缀。普通局部末端速度和加速度自由，前端折点用作软跟踪参考；真正目标端要求精确到达并停止；普通计划另带停车位置自由的制动备用段，连续接续失败时才实际执行。使用机器人位移距离、沿边形状和障碍风险、ChooseParent 与 Rewire；连续环境和车间安全使用共同凸区域与定向分离约束。仿射形状边界在每段采用局部凸内近似。

```bash
MPLCONFIGDIR=/tmp/mpl-formation OPENBLAS_NUM_THREADS=1 /home/eai/projects/env-rebuilt/bin/python scripts/experiments/run_local_affine.py --scene random_pillars --obstacle-count 50 --mode affine --iterations 250 --execution-dt .02 --perception-period .2 --replan-period 1 --max-time 500 --progress --output results/local_affine/run
```

`--mode uniform|affine` 选择统一缩放或一般仿射；地图与初终质心默认分别为 50×40 m 和 (5,20)→(45,20)，三角六车间距 3.5 m、方柱宽 0.3 m 是自选假设。`trajectory.npz` 保存执行状态、即时速度/角速度、连续曲线控制点、区域约束与感知记录；`plans.json` 保存各轮树、空间安全区域和各段执行区间。

执行、感知和规划在同一同步仿真循环按各自仿真时间周期调度；规划计算耗时不会推进仿真时间。曲线段时长按执行步长取整。旧参数 `--max-steps`、`--replan-steps` 显式使用时分别按执行步长换算并覆盖 `--max-time`、`--replan-period`。

本轮对比显式采用 1 秒重规划，非终点主轨迹的估计时域不少于两个重规划周期。时长由路径与初始运动状态估计并做必要条件检查，并非完整的时间分配优化。制动备用段在主轨迹求解后独立求解和验收。

差速车状态通过共享平坦输出曲线及导数解析恢复，静止时可原地转向；这不是包含执行器和跟踪误差的闭环控制实验。初版存在多次停走，不构成实时性能结论。速度与加速度使用 Bézier 导数控制点约束，角速度通过顺序凸化修正并以全段多项式数值极值检查验收，修正使用自适应极值位置约束，直接数值装配至 Clarabel；具体轨迹仍需独立验收；有限预算局部搜索不保证到达。

`metrics.json` 的 `timings_seconds` 分开记录感知、SFC、前端、后端和执行计算耗时；`simulation_duration_seconds` 是机器人运动的仿真时长。前端缓存节点几何并按代价下界剪枝。后端通过 `trajectory_template.py` 按段数和各段实际约束行数缓存参数化凸问题，跨轮更新路径、SFC 和初始状态，并为每次求解设置选定的时长；缓存不会改变对应优化问题的目标与约束，不补齐约束行。当前主目标包含前端位置与参考速度的积分跟踪；备用段仍仅惩罚加速度。缓存归各次仿真独立所有，日志中的 `template_cache_hit` 记录复用情况。
