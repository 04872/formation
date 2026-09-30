# Tube-RRT 可视化结果索引

本文件由 `scripts/visualize_tube_rrt.py` 在每次运行后自动重新生成，请勿手动编辑。

目录结构：`results/tube_rrt/<地图>_seed<seed>/<变体>/`，变体名依次由以下部分组成：编队名；`x<k>`（槽位整体放大 k 倍，k=1 时省略）；`first_goal`（首次连到目标即停止）或 `anytime_it<N>`（跑满 N 次迭代，保留代价最小的目标节点）；`wm<w>`（J_margin 权重）；`fixedstep`（关闭步长回退）。编队名后的 `cell1` / `cell2` 表示一阶 / 二阶 chart cell，`cellP` 表示 polyhedral cell + region/uniform 混合 RRT*（默认 p_region 首解前 0.5、之后 0.2；`p<p>` 为常数 p_region，`pexp<max>-<min>-k<k>` 为指数衰减调度，`p<before>-<after>` 为非默认切换值），没有该标记的是原来的 orientation-sliced cell。每个运行目录下的 `run.md` 是可直接阅读的运行报告。

“夹障碍节点”指障碍中心落在机器人凸包内的路径节点数，用来判断规划是否利用了“障碍从机器人之间穿过”。

“可穿过障碍”指半径小于“相邻机器人间隙允许的障碍半径上限”的障碍个数；为 0 时规划器只能绕行。

| 运行 | cell | 成功 | 节点 | 首次到达迭代 | 代价 首次→最终 | d_G 长度 | 最小 guarded clearance | 可穿过障碍 / 上限 m | 夹障碍节点 / 路径节点 | 规划耗时 s | 运行时间 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [post_fence_seed7/square_cell1_x2_anytime_it2500](post_fence_seed7/square_cell1_x2_anytime_it2500/run.md) | first_order | 是 | 2343 | 113 | 12.074 → 10.806 | 10.800 | 0.065 | 7/7 / 0.327 | 9 / 35 | 0.32 | 2026-09-28 17:05:10 |
| [post_fence_seed7/square_cell2_x2_anytime_it2500](post_fence_seed7/square_cell2_x2_anytime_it2500/run.md) | second_order | 是 | 2338 | 113 | 12.045 → 10.802 | 10.802 | 0.065 | 7/7 / 0.327 | 9 / 34 | 0.50 | 2026-09-28 17:04:46 |
| [post_fence_seed7/square_cellP_x2_anytime_it2500](post_fence_seed7/square_cellP_x2_anytime_it2500/run.md) | polyhedral | 是 | 2084 | 45 | 12.431 → 11.364 | 11.364 | 0.150 | 7/7 / 0.327 | 4 / 18 | 2.81 | 2026-09-30 10:37:02 |
| [post_fence_seed7/square_x2_anytime_it2500](post_fence_seed7/square_x2_anytime_it2500/run.md) | orientation | 是 | 2315 | 76 | 11.649 → 11.363 | 11.363 | 0.124 | 7/7 / 0.327 | 12 / 52 | 3.91 | 2026-09-27 14:24:10 |
| [random_circles_seed7/square_cell1_x2_anytime_it2500](random_circles_seed7/square_cell1_x2_anytime_it2500/run.md) | first_order | 是 | 2196 | 155 | 13.676 → 12.040 | 12.031 | 0.218 | 12/12 / 0.327 | 0 / 27 | 0.33 | 2026-09-28 17:32:30 |
| [random_circles_seed7/square_cell2_x2_anytime_it2500](random_circles_seed7/square_cell2_x2_anytime_it2500/run.md) | second_order | 是 | 2115 | 155 | 13.688 → 12.031 | 12.031 | 0.218 | 12/12 / 0.327 | 0 / 27 | 0.51 | 2026-09-28 17:30:43 |
| [random_circles_seed7/square_cellP_x2_anytime_it2500](random_circles_seed7/square_cellP_x2_anytime_it2500/run.md) | polyhedral | 是 | 1963 | 51 | 11.225 → 11.225 | 11.225 | 0.156 | 12/12 / 0.327 | 0 / 12 | 2.96 | 2026-09-30 10:37:02 |
| [random_circles_seed7/square_x2_anytime_it2500](random_circles_seed7/square_x2_anytime_it2500/run.md) | orientation | 是 | 2143 | 236 | 12.807 → 12.406 | 12.406 | 0.150 | 12/12 / 0.327 | 0 / 58 | 0.63 | 2026-09-28 17:58:01 |
| [single_post_seed7/square_cell1_x2_anytime_it2500](single_post_seed7/square_cell1_x2_anytime_it2500/run.md) | first_order | 是 | 2352 | 164 | 13.418 → 11.993 | 11.994 | 0.179 | 1/1 / 0.327 | 0 / 29 | 0.32 | 2026-09-28 17:05:03 |
| [single_post_seed7/square_cell2_x2_anytime_it2500](single_post_seed7/square_cell2_x2_anytime_it2500/run.md) | second_order | 是 | 2356 | 164 | 13.421 → 11.996 | 11.996 | 0.179 | 1/1 / 0.327 | 0 / 28 | 0.50 | 2026-09-28 17:04:39 |
| [single_post_seed7/square_cellP_x2_anytime_it2500](single_post_seed7/square_cellP_x2_anytime_it2500/run.md) | polyhedral | 是 | 2011 | 44 | 11.666 → 11.582 | 11.582 | 0.067 | 1/1 / 0.327 | 0 / 15 | 2.45 | 2026-09-30 10:37:01 |
| [single_post_seed7/square_x2_anytime_it2500](single_post_seed7/square_x2_anytime_it2500/run.md) | orientation | 是 | 2374 | 113 | 12.178 → 11.089 | 11.089 | 0.012 | 1/1 / 0.327 | 14 / 82 | 3.58 | 2026-09-27 10:06:36 |
