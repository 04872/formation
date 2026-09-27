# Tube-RRT 可视化结果索引

本文件由 `scripts/visualize_tube_rrt.py` 在每次运行后自动重新生成，请勿手动编辑。

目录结构：`results/tube_rrt/<地图>_seed<seed>/<变体>/`，变体名依次由以下部分组成：编队名；`x<k>`（槽位整体放大 k 倍，k=1 时省略）；`first_goal`（首次连到目标即停止）或 `anytime_it<N>`（跑满 N 次迭代，保留代价最小的目标节点）；`wm<w>`（J_margin 权重）；`fixedstep`（关闭步长回退）。每个运行目录下的 `run.md` 是可直接阅读的运行报告。

“夹障碍节点”指障碍中心落在机器人凸包内的路径节点数，用来判断规划是否利用了“障碍从机器人之间穿过”。

“可穿过障碍”指半径小于“相邻机器人间隙允许的障碍半径上限”的障碍个数；为 0 时规划器只能绕行。

| 运行 | 成功 | 节点 | 首次到达迭代 | 代价 首次→最终 | d_G 长度 | 最小 guarded clearance | 可穿过障碍 / 上限 m | 夹障碍节点 / 路径节点 | 规划耗时 s | 运行时间 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [post_fence_seed7/square_first_goal](post_fence_seed7/square_first_goal/run.md) | 否 | 1132 | - | - → - | - | 0.002 | 0/7 / 0.077 | 0 / 0 | 1.84 | 2026-09-26 16:10:38 |
| [post_fence_seed7/square_x2_anytime_it2500](post_fence_seed7/square_x2_anytime_it2500/run.md) | 是 | 2315 | 76 | 11.649 → 11.363 | 11.363 | 0.124 | 7/7 / 0.327 | 12 / 52 | 3.91 | 2026-09-27 14:24:10 |
| [post_fence_seed7/square_x2_first_goal](post_fence_seed7/square_x2_first_goal/run.md) | 是 | 77 | 76 | 11.240 → 11.240 | 11.240 | 0.101 | 7/7 / 0.327 | 4 / 17 | 0.02 | 2026-09-26 16:17:41 |
| [random_circles_seed7/square_anytime_it2500](random_circles_seed7/square_anytime_it2500/run.md) | 是 | 2336 | 126 | 12.434 → 11.281 | 11.281 | 0.318 | 0/12 / 0.077 | 0 / 14 | 2.58 | 2026-09-26 12:03:31 |
| [random_circles_seed7/square_first_goal](random_circles_seed7/square_first_goal/run.md) | 是 | 104 | 126 | 12.434 → 12.434 | 12.434 | 0.096 | 0/12 / 0.077 | 0 / 20 | 0.06 | 2026-09-26 11:40:49 |
| [random_circles_seed7/square_x2_anytime_it2500](random_circles_seed7/square_x2_anytime_it2500/run.md) | 是 | 2143 | 236 | 12.807 → 12.406 | 12.406 | 0.150 | 12/12 / 0.327 | 0 / 58 | 4.64 | 2026-09-27 14:28:17 |
| [single_post_seed7/square_x2_anytime_it2500](single_post_seed7/square_x2_anytime_it2500/run.md) | 是 | 2374 | 113 | 12.178 → 11.089 | 11.089 | 0.012 | 1/1 / 0.327 | 14 / 82 | 3.58 | 2026-09-27 10:06:36 |
