# Tube-RRT 运行记录：single_post / seed 7 / square_cell2_x2_anytime_it2500

- 运行时间：2026-09-28 17:04:39
- 代码版本：`be68102 (有未提交改动)`
- 结果目录：`results/tube_rrt/single_post_seed7/square_cell2_x2_anytime_it2500`

复现命令（在仓库根目录执行）：

```bash
MPLBACKEND=Agg ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py --cell second_order --anytime --map single_post --slot-scale 2 --progress-interval 0
```

## 结果

| 项目 | 值 |
| --- | --- |
| 是否成功 | 是 |
| 执行迭代数 | 2500 / 2500 |
| 首次连到目标的迭代 | 164 |
| 树节点数（含目标节点） | 2356（目标节点 7） |
| 被拒绝：碰撞 / cell 不重叠 | 103 / 49 |
| rewire 次数 | 1743 |
| 步长回退后才接受的节点数 | 663 |
| 障碍夹在机器人之间的节点：树 / 路径 | 41 / 0（路线共 28 个航点，含 portal） |
| 路径代价（含 J_margin）：首次 → 最终 | 13.421 → 11.996 |
| 路线长度 ||.||_F | 11.996 |
| 认证路线上稠密采样的最小 guarded clearance（<0 表示碰撞） | 0.179 |
| overlap 判定：总数 / 快速拒绝 / 快速接受 / SOCP（接受） | 12049 / 6077 / 5275 / 697（522） |

## 配置

| 参数 | 值 |
| --- | --- |
| cell | 二阶 cell：max_i (||u + J a_i phi|| + ||r_i|| phi^2 / 2) < d_k，d_k = eta d_obs；overlap 用 SOCP（先做必要条件快速拒绝和候选 portal 快速接受） |
| 地图 | single_post, seed 7 |
| 编队 | square × 2（R_F = 0.707 m，相邻机器人最小间距 1.000 m） |
| 机器人半径 / 安全余量 | 0.113 m（TurtleBot3 Burger 外接圆） / 0.060 m |
| 障碍半径 | 0.150–0.150 m（设置：地图默认，共 1 个） |
| 能从相邻机器人之间穿过的障碍半径上限 | 0.327 m（= 最宽相邻间距/2 − 机器人半径 − 安全余量；小于它的障碍 1/1 个） |
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
| cell_model | second_order |
| cell_eta | 0.98 |
| route_check_step | 0.02 |

## 耗时

| 阶段 | 秒 |
| --- | --- |
| startup_imports | 0.207 |
| map_build | 0.083 |
| tube_rrt | 0.498 |
| plot_build | 1.109 |
| save | 1.214 |
| total | 3.112 |

## 图

### overview.png

总览：搜索树、joint tube、编队投影、tube 宽度四宫格

![overview](overview.png)

### 1_tree.png

最终搜索树 (x, y) 投影，颜色 = 节点插入顺序，短线 = yaw；叠加被拒绝的 steer 点、rewire 移除的边，粉色圈 = 障碍夹在机器人之间的节点

![tree](1_tree.png)

### 2_growth.png

由 trace 回放的树生长快照：sample、nearest、q_new 及其 cell 在节点 yaw 处的 xy 截面

![growth](2_growth.png)

### 3_tube.png

joint tube：路径 cell 的节点 yaw 截面与 portal、(x, y, theta) 中的 cell 形状、沿认证路线的稠密 clearance、最紧 portal 处两个 cell 的固定 yaw 截面（圆盘交集）

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

## 其他文件

- `path.csv`：认证路线 q_0 -> portal -> q_1 -> ... 的航点（cx, cy, theta, guarded_clearance）及各机器人投影坐标。
- `summary.json`：本次运行的机器可读摘要，`results/tube_rrt/README.md` 索引由它生成。

## 控制台输出

```text
timing startup_imports=0.207s
timing map_build=0.083s
geometry robot_radius=0.113 safety_margin=0.060 pass_through_bound=0.327 obstacle_radius=0.150-0.150 passable_obstacles=1/1
planning...
timing tube_rrt=0.498s
planning result: success=True iterations=2500 nodes=2356 path_nodes=28 path_cost=11.996 route_min_clearance=0.179
overlap overlap_calls=12049 quick_reject=6077 quick_accept=5275 socp_calls=697 socp_accept=522
timing plot_build=1.109s
timing save=1.214s total=3.112s
saved results/tube_rrt/single_post_seed7/square_cell2_x2_anytime_it2500/
```
