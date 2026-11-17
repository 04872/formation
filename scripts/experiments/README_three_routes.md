# 三路线同频对比

入口：`compare_formation_routes.py`。结果：`results/three_routes/matched_rates/`，分为
`paper_uniform`（统一缩放＋MPPI）、`paper_affine`（一般仿射＋MPPI）、
`rrt_affine`（局部 RRT*＋集中式 Bézier 优化），每种包含 `seed_0/1`。

三条均采用差速运动学，执行 50 Hz、感知 5 Hz、前端和后端各 1 Hz。
地图为同一组 50 个随机方柱，地图 seed=1，六机器人三角队形间距 3.5 m。
共同到达条件：每个机器人目标误差 <0.4 m 且速度 <0.1 m/s；最长模拟 500 s。
MPPI 保留 0.2 s 预测节点与 7 s 时域，每次优化后执行五个控制节点，各自按 0.02 s 积分。
集中优化执行解析轨迹，末段承诺和尾段恢复可影响实际重规划间隔。
这是离线同步仿真，计算耗时不推进模拟时钟，不代表实时调度验证。

工程上共用观测地图、FIRI、精确距离查询；批量候选采样、MPPI 距离查询及
RRT/凸优化构图各自减少重复开销。前两者原采样预算保持不变，第三条使用 250 次 RRT 迭代，
因此不是等计算预算或单变量消融。等价性和耗时记录位于 `results/three_routes/engineering_before/`。

运行示例（仓库根目录）：

```bash
formation/paper_formation/native/build_affine_sampler.sh
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-formation /home/eai/projects/env-rebuilt/bin/python scripts/experiments/compare_formation_routes.py --route paper_uniform --seed 0 --progress
```

改变 `--route` 和 `--seed` 可运行其他五组。没有新增 Python 包；原生采样器链接现有 NumPy 随机库，
在新机器或 NumPy 更新后重新编译。

每组保存配置、轨迹和指标。纸面路线用 `scripts/paper_formation/audit_paper_formation.py trajectory.npz --regions --output audit.json`，
RRT 路线用 `scripts/diagnostics/audit_local_affine.py <结果目录>` 独立复核。
六组完成后运行 `scripts/diagnostics/summarize_formation_routes.py results/three_routes/matched_rates` 汇总，
再运行 `scripts/visualization/render_three_routes.py` 生成同一模拟时钟下的并排视频及轨迹图。
两次重复不足以推断普遍成功率；超时和 SFC 构建失败均保留在结果中。
