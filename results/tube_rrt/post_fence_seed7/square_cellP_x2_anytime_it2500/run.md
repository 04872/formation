# Tube-RRT 运行记录：post_fence / seed 7 / square_cellP_x2_anytime_it2500

- 运行时间：2026-09-29 18:32:40
- 代码版本：`1d6ec45 (有未提交改动)`
- 结果目录：`results/tube_rrt/post_fence_seed7/square_cellP_x2_anytime_it2500`

复现命令（在仓库根目录执行）：

```bash
MPLBACKEND=Agg ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py --map post_fence --slot-scale 2 --anytime --cell polyhedral --progress-interval 0
```

## 结果

| 项目 | 值 |
| --- | --- |
| 是否成功 | 是 |
| 执行迭代数 | 2500 / 2500 |
| 首次连到目标的迭代 | 65 |
| 树节点数（含目标节点） | 1051（目标节点 3） |
| 被拒绝：碰撞 / 安全球不重叠 | 0 / 0 |
| rewire 次数 | 330 |
| 步长回退后才接受的节点数 | 894 |
| 障碍夹在机器人之间的节点：树 / 路径 | 161 / 3（路径共 20 个节点） |
| 路径代价（含 J_margin）：首次 → 最终 | 14.541 → 11.978 |
| 路径 d_G 长度 | 11.978 |
| 认证路线上稠密采样的最小 guarded clearance（<0 表示碰撞） | 0.118 |
| overlap 判定：总数 / 快速拒绝 / 快速接受 / LP（接受） | 9538 / 1 / 7175 / 2362（1377） |
| 被拒绝：冗余 cell（rho_new 过小） | 1453 |
| 采样来源 frontier / uniform：迭代数 | 2141 / 359 |
| 采样来源 frontier / uniform：接受节点数 | 688 / 359 |
| frontier 候选：生成 / 被覆盖 / 多次失败丢弃 / 结束时仍 exposed | 32494 / 14384 / 0 / 18110 |
| 构造 cell 数 / 平均 active pair 数 | 7165 / 3.60 |

## 配置

| 参数 | 值 |
| --- | --- |
| cell | polyhedral cell + frontier 搜索：active robot-obstacle 支撑平面 n^T dc - rho|dtheta| >= -(d - d_s) 与 validity guard ||dc|| + rho|dtheta| <= d_inactive - d_s（内接多边形）之交；从 exposed frontier 按 S = alpha l + beta U + gamma G 采样 (i, u, dtheta)，按 J_expand 选 seed，混合少量 uniform SE(2) 采样 |
| 地图 | post_fence, seed 7 |
| 编队 | square × 2（R_F = 0.707 m，相邻机器人最小间距 1.000 m） |
| 机器人半径 / 安全余量 | 0.113 m（TurtleBot3 Burger 外接圆） / 0.060 m |
| 障碍半径 | 0.080–0.080 m（设置：地图默认，共 7 个） |
| 能从相邻机器人之间穿过的障碍半径上限 | 0.327 m（= 最宽相邻间距/2 − 机器人半径 − 安全余量；小于它的障碍 7/7 个） |
| max_iterations | 2500 |
| metric_step | 0.45 |
| neighbor_radius | 1.2 |
| goal_bias | 0.12 |
| goal_connect_distance | 0.8 |
| seed | 7 |
| safety_margin | 0.0 |
| progress_interval | 0 |
| stop_on_first_goal | False |
| step_backoff | True |
| min_metric_step | 0.02 |
| margin_weight | 0.0 |
| yaw_slices | 16 |
| cell_eta | 0.98 |
| yaw_slice_count | None |
| cell_shrink | None |
| cell_model | polyhedral |
| route_check_step | 0.02 |
| frontier.safety_distance | 0.02 |
| frontier.active_range | 1.0 |
| frontier.max_active_per_robot | 3 |
| frontier.max_extent | 1.5 |
| frontier.yaw_limit | 1.5707963267948966 |
| frontier.guard_facets | 12 |
| frontier.cell_samples | 128 |
| frontier.yaw_slice_fractions | (0.0, -0.5, 0.5, -0.9, 0.9) |
| frontier.frontier_spacing | 0.3 |
| frontier.probe_step | 0.25 |
| frontier.probe_count | 3 |
| frontier.uniform_probability | 0.15 |
| frontier.score_weights | (1.0, 1.0, 1.0) |
| frontier.score_temperature | 0.15 |
| frontier.length_ref | 1.0 |
| frontier.expand_weights | (1.0, 0.5, 0.5) |
| frontier.overlap_band | (0.1, 0.5) |
| frontier.min_new_ratio | 0.05 |
| frontier.uniform_min_new_ratio | 0.0 |
| frontier.step_scales | (0.8, 1.2, 1.6) |
| frontier.max_candidate_failures | 3 |
| frontier.max_parent_candidates | 12 |

## 耗时

| 阶段 | 秒 |
| --- | --- |
| startup_imports | 0.223 |
| map_build | 0.082 |
| tube_rrt | 5.031 |
| plot_build | 1.200 |
| save | 1.278 |
| total | 7.816 |

## 图

### overview.png

总览：搜索树、joint tube、编队投影、tube 宽度四宫格

![overview](overview.png)

### 1_tree.png

最终搜索树 (x, y) 投影，颜色 = 节点插入顺序，短线 = yaw；叠加被拒绝的 steer 点、rewire 移除的边，粉色圈 = 障碍夹在机器人之间的节点

![tree](1_tree.png)

### 2_growth.png

由 trace 回放的树生长快照：sample、nearest、q_new 及其安全 cell anchor-yaw 截面。

![growth](2_growth.png)

### 3_tube.png

joint tube：路径 polyhedral cell 的 dtheta = 0 截面 P_i(0) 与 portal、(x, y, theta) 中叠起来的 yaw 截面多边形、沿认证路线的稠密 clearance、最紧 portal 处两个 cell 在 portal yaw 下的截面（实线边 = active 障碍支撑平面，虚线边 = validity guard）

![tube](3_tube.png)

### 4_formation.png

沿路径等间距采样的编队姿态：每个姿态一种颜色并编号，轮廓线连接各机器人（粉色填充 = 障碍夹在机器人之间）；机器人轨迹、局部 guarded clearance 圆

![formation](4_formation.png)

### 5_formation_frames.png

关键帧放大：障碍夹在机器人之间（或瓶颈）附近的连续 tube 节点；每个子图以“上一节点 + 当前节点”为中心、比例尺相同，虚线为上一节点姿态，黑箭头为中心位移

![frames](5_formation_frames.png)

### 6_convergence.png

最优路径代价随迭代下降曲线，以及节点 / 拒绝 / rewire 累计数

![convergence](6_convergence.png)

### 7_frontier.png

（仅 polyhedral）certified union：全部 cell 的节点 yaw 截面、仍 exposed 的 frontier 候选（颜色 = 采样分数 S）；新 cell 的 rho_new / rho_overlap 分布；frontier 与 uniform 两种采样来源的累计接受节点数

![frontier](7_frontier.png)

## 其他文件

- `path.csv`：展开后的联合安全路线（cx, cy, theta, guarded_clearance）及各机器人投影坐标。
- `summary.json`：本次运行的机器可读摘要，`results/tube_rrt/README.md` 索引由它生成。

## 控制台输出

```text
timing startup_imports=0.223s
timing map_build=0.082s
geometry robot_radius=0.113 safety_margin=0.060 pass_through_bound=0.327 obstacle_radius=0.080-0.080 passable_obstacles=7/7
planning...
timing tube_rrt=5.031s
planning result: success=True iterations=2500 nodes=1051 path_nodes=20 path_cost=11.978 min_guarded_clearance=0.118
overlap cells_built=7165 active_pairs=25799 overlap_calls=9538 quick_reject=1 quick_accept=7175 lp_calls=2362 lp_accept=1377 frontier_iterations=2141 uniform_iterations=359 frontier_nodes=688 uniform_nodes=359 frontier_candidates=32494 frontier_covered=14384 frontier_dropped=0 rejected_redundant=1453 frontier_alive=18110
timing plot_build=1.200s
timing save=1.278s total=7.816s
saved results/tube_rrt/post_fence_seed7/square_cellP_x2_anytime_it2500/
```
