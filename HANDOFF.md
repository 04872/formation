# HANDOFF

## 交接范围与当前目标

本仓库当前交付的是 rigid-formation Joint Tube-RRT 全局折线路径：把固定槽位的整队作为一个联合系统，在 SE(2) 联合构型空间中搜索一条带局部安全裕度的可行路径，并将联合路径投影为各机器人路径。当前输出仍是未平滑的折线联合路径；不包含全局曲线路径优化、portal 优化、B-spline/Bezier、动力学、控制器、MPC 或机器人闭环运动。

设计背景可对照普通 `method/编队全局路径规划.md`，但当前实现范围以源码为准。

## 关键文件图谱

- `formation/tube_rrt.py:12-20`：`TubeRRTConfig`；`formation/tube_rrt.py:24-39`：节点和结果结构；`formation/tube_rrt.py:43-58`：槽位刚体变换、最短角插值。
- `formation/tube_rrt.py:60-115`：`TubeRRTPlanner`、度量、精确 clearance、安全半径和严格 tube overlap。
- `formation/tube_rrt.py:116-235`：采样、steering、nearest、父节点选择、rewiring、后代 cost 更新、goal 连接与路径输出；`formation/tube_rrt.py:238-244`：机器人路径投影。
- `formation/map_config.py:16-120`：地图、随机圆障碍和规划配置；默认随机圆地图 seed 为 7。
- `formation/map_builder.py:19-31`：地图类型分派；`formation/map_builder.py:57-91`：距离场、膨胀栅格和 `MapData` 构造；`formation/map_builder.py:116-126`：圆形栅格化；`formation/map_builder.py:238-272`：随机圆生成。
- `formation/types.py:12-39`：`Pose2D`、`MapData`；`formation/types.py:92-101`：`FormationSpec`；`formation/types.py:224-226`：yaw wrap。
- `formation/__init__.py:25-31`、`formation/__init__.py:62-120`：Tube-RRT API 和相关类型的公开导出。
- `scripts/visualize_tube_rrt.py:30-116`：CLI、规划、绘图、计时和保存。
- `tests/test_tube_rrt.py:21-124`：刚体变换、度量、圆障碍/边界 clearance、间隙穿行、严格 overlap、进度输出、rewire/cost、确定性和起点碰撞测试。
- `tests/test_maps.py:22-145`：全部地图字段、随机圆确定性/边界、各类走廊和起终点验证。
- `formation/distance_field.py:45-90`：二维 occupancy 的可分离平方欧氏距离变换及膨胀阈值。

## 当前算法与安全语义

联合状态是 `q=(cx, cy, theta)`。固定 formation slots 位于质心坐标系，`transform_slots` 用同一个二维旋转和平移得到每个机器人中心：`p_i = c + R(theta) r_i`（实现见 `formation/tube_rrt.py:43-49`）。起点姿态固定为脚本中的 `Pose2D(*map_data.start_xy, 0.0)`（`scripts/visualize_tube_rrt.py:46-50`）；终点中心精确设置为 `goal_xy`，yaw 继承到达节点的 yaw 且自由（`formation/tube_rrt.py:199-205`）。

度量为

`d_G(q_a,q_b) = ||c_a-c_b||_2 + formation_radius * |wrapped(theta_b-theta_a)|`，

见 `formation/tube_rrt.py:86-88`。其中 `formation_radius` 是 slots 到质心的最大距离。每个联合姿态的 clearance 逐机器人计算：对每个 slot 中心分别检查地图四条边界和每个圆形障碍物，再减机器人半径和安全 margin（`formation/tube_rrt.py:90-100`）。因此绝不使用整队外包圆、包围盒或凸包做碰撞膨胀；障碍物可以安全通过槽位之间的间隙。地图边界同样计入 clearance。定义 `rho=max(0, clearance)`（`formation/tube_rrt.py:102-103`）。当前 `TubeRRTPlanner` 只支持 `type == "circle"` 的障碍 primitive，遇到其他 primitive 会在构造时拒绝（`formation/tube_rrt.py:82-84`）。

节点安全球使用上述 `d_G` 和 `rho`。相邻节点严格按 `d_G < rho_a + rho_b` 判断 overlap（`formation/tube_rrt.py:105-114`），不是小于等于。该度量上界约束每个槽位中心的世界位移，所以 `rho` 对应该联合姿态周围的认证安全度量球。位置线性插值与 yaw 最短角插值构成此度量下的等速测地线；严格重叠保证连接测地线上的每一点至少落在两个安全球之一，因此球链可连续安全连接。这不是“仅近似安全”。测试还验证了障碍物位于槽位之间时不会被错误的 formation hull 膨胀挡住（`tests/test_tube_rrt.py:56-70`）。

搜索过程是 seeded SE(2) sampling，带 goal bias；对最近节点按上述 metric steering，检查新节点 clearance 和与最近节点的 tube overlap。

**步长回退（`step_backoff=True`，默认）**：先用 `metric_step`（0.45）尝试；失败则步长减半重试，最小退到 `0.9 * rho_nearest`。clearance 对 `d_G` 是 1-Lipschitz 的，步长小于 `rho_nearest` 时新节点必然安全且与最近球严格重叠，所以最后一次尝试必定成功，除非它小于 `min_metric_step`（0.02）。关闭回退时，固定步长加重叠条件隐含要求相邻节点平均 `rho` 大于约 0.225，窄通道（包括“障碍从机器人之间穿过”的构型）无法进入，结果看起来像把障碍额外膨胀后做质点规划。对照：`post_fence` 地图上的默认 square，固定步长失败，回退后成功。

**J_margin（`margin_weight`，默认 0）**：边代价为 `d_G * (1 + w / min(rho_a, rho_b))`，所有 RRT* 代价（父节点选择、rewire、后代更新、goal）统一走 `edge_cost`，w=0 时与纯 `d_G` 长度一致。seed=7 square、w=0.1 时，瓶颈 rho 从 0.341 升到 0.499，d_G 长度从 11.31 增到 11.44。

然后在邻域内做 parent selection（`formation/tube_rrt.py:179-188`），再做 rewiring，并递归更新 descendant cost（`formation/tube_rrt.py:191-198`、`formation/tube_rrt.py:133-144`）。到目标中心足够近时，创建中心精确为目标、yaw 继承到达节点的 goal node。`TubeRRTConfig.stop_on_first_goal=True`（库默认）时首次连到目标即返回；设为 `False` 时为 anytime 模式：跑满 `max_iterations`，只在新 goal node 代价低于当前最优时才加入，rewire 可继续降低已有 goal node 代价，最终输出代价最小的 goal node 路径。结果保留搜索树，并输出未平滑的 joint 折线路径及对应 radii，另含 `iterations`、`first_goal_iteration`、`path_cost` 和最优代价下降记录 `cost_history`。`progress_interval` 大于 0 时输出进度，否则默认静默。

## 可视化与计时

`scripts/visualize_tube_rrt.py` 支持：

- `--out-dir DIR`：结果根目录，默认 `results/tube_rrt`（`results/` 已被 gitignore）。
- `--map {random_circles,post_fence,single_post}`，默认 `random_circles`。
- `--formation {square,column,horizontal_line,t_shape}`；`--slot-scale K` 把槽位整体放大 K 倍（默认 1）。
- `--robot-radius R`：机器人外接圆半径，默认 0.113 m（TurtleBot3 Burger 底盘 138×178 mm 的外接圆半径）；同时决定默认槽位间距（`FormationLibrary.build_default(R)`）。
- `--safety-margin M`：障碍膨胀余量，默认 0.06 m。
- `--obstacle-radius {R | MIN:MAX | auto}`：障碍半径。`random_circles` 取 R 或区间 [MIN, MAX]；柱子地图只接受单值。`auto` 按“可穿过上限” b 自动取值：`random_circles` 取 (0.3b, 0.8b)，柱子取 0.5b；b ≤ 0 时报错。不传时保持地图默认值。
- `--obstacle-count N`：`random_circles` 的障碍个数。
- `--seed INT`：规划器 seed，同时是 `random_circles` 的地图 seed。
- `--iterations INT`：迭代预算，默认 2500；脚本默认 anytime 模式跑满预算。
- `--first-goal`：首次连到目标即停止。
- `--margin-weight W`：J_margin 权重，默认 0。
- `--no-step-backoff`：关闭步长回退（旧的固定步长行为）。
- `--progress-interval INT`，非负，默认 500；传 0 关闭搜索进度输出。
- `--show`：保存后再交互式显示全部图。

**测试地图与几何前提**：可穿过障碍半径上限 `b = max(凸包边界上相邻机器人间距的一半) − robot_radius − safety_margin`，半径小于 b 的障碍才可能从机器人之间穿过。脚本在控制台 `geometry` 行、`run.md` 和索引中给出 b 与可穿过障碍个数；个数为 0 时打印提示。Burger 半径 0.113、margin 0.06 下：

| 编队 | b（×1） | b（×2） |
| --- | --- | --- |
| square / column / horizontal_line | 0.077 | 0.327 |
| t_shape | 0.181 | 0.534 |

`random_circles` 默认障碍半径 0.12–0.28、`single_post` 默认 0.15、`post_fence` 默认 0.08，都大于 ×1 下的 0.077，所以默认场景只能绕行（这是正确行为）。要展示障碍从机器人之间穿过，可以用 `--obstacle-radius auto` 缩小障碍，或者用 `--slot-scale 2` 放大编队：

- `post_fence`（`PostFenceConfig`）：x=0 处一排柱子，间距 1.1 m，半径 0.08 m。配合 `--slot-scale 2` 时整队无法从两柱之间通过，只能让柱子从机器人之间穿过。
- `single_post`（`SinglePostConfig`）：起终点连线上只有一根半径 0.15 m 的柱子。`--slot-scale 2` 时最短路径骑跨柱子。

每次运行保存到 `<out-dir>/<map>_seed<seed>/<变体>/`，变体名依次由以下部分组成：编队名；`x<K>`；`r<R>`、`sm<M>`（非默认机器人半径 / 余量）；`obs<spec>`、`n<N>`（障碍半径 / 个数）；`anytime_it<N>` 或 `first_goal`；`wm<W>`；`fixedstep`。同参数重跑会覆盖同一目录（搜索是确定性的）。目录内容：

- `run.md`：给人阅读的运行报告，含复现命令、代码版本、结果与配置表、耗时、全部图和完整控制台输出。
- `overview.png`、`1_tree.png`、`2_growth.png`、`3_tube.png`、`4_formation.png`、`5_formation_frames.png`、`6_convergence.png`。
- `path.csv`：联合路径节点 (cx, cy, theta, rho) 和各机器人投影坐标；`summary.json`：机器可读摘要。

每次运行后自动重新生成 `<out-dir>/README.md`，汇总所有运行（节点数、首次/最终代价、d_G 长度、瓶颈、障碍夹在机器人之间的路径节点数、耗时），并链接到各自的 `run.md`。“障碍夹在机器人之间”指障碍中心落在机器人中心的凸包内；树图用粉色圈标出这类节点，编队图用粉色填充这类姿态。

脚本以 `TubeRRTConfig(record_trace=True)` 规划，`TubeRRTResult.trace` 逐迭代记录采样、最近节点、steer 结果、`rho`、状态（`collision`/`no_overlap`/`added`/`goal`）和 rewire；默认关闭，不影响搜索与随机序列。各图含义：

- `tree`：最终树的 (x, y) 投影，边和节点按插入顺序着色，节点短线表示 yaw；叠加被拒绝的 steer 点（碰撞 / 不重叠）和被 rewire 移除的旧边。节点多于 600 时不画 yaw 短线，rewire 多于 300 时只在图例中给出数量。
- `growth`：由 trace 回放的 6 个快照（首次到达目标前 3 张、首次到达、之后 1 张、最终），标出当次 sample、nearest、q_new 及其安全球 xy 截面和当时的最优路径。
- `convergence`：最优 goal 代价随迭代的阶梯下降曲线，以及接受节点 / 碰撞拒绝 / 不重叠拒绝 / rewire 的累计数。
- `tube`：路径节点安全球在固定 theta 下的 xy 截面（半径 `rho_k` 的圆盘）、(x, y, theta) 中的双锥 `||dc|| + R_F|dtheta| < rho`、`rho_k` 沿路径曲线及瓶颈、相邻节点 `d_G` 与 `rho_k + rho_{k+1}` 的严格重叠检查。
- `formation`：视野裁剪到路径附近。沿联合路径按 `平移 + R_F·转角` 等间距采样 10 个姿态（在测地线上插值，不局限于 tube 节点），每个姿态一种颜色（turbo 色表）并在中心编号；轮廓线连接各机器人，粉色填充表示障碍夹在机器人之间。机器人轨迹按每条边稠密插值绘制，瓶颈节点处画出半径 `r + margin + rho` 的圆（与最近障碍相切）；下方为 theta 沿路径曲线。
- `formation_frames`：6 个关键帧，取障碍夹在机器人之间的路径节点区间（没有时取瓶颈节点附近）。每个子图以“上一节点 + 当前节点”为中心、比例尺相同；当前姿态为蓝色轮廓，上一节点为灰色虚线，黑箭头为中心位移，被夹住的障碍描粉色边；标题给出 theta、rho 和这一步的 d_G。
- 总览：tree、tube xy、formation、rho 曲线四宫格。

脚本计时输出包括 `startup_imports`、`map_build`、`tube_rrt`、`plot_build`、`save`、`total`，同时写入 `run.md` 和 `summary.json`。

单次实测参考（Burger 半径 0.113、margin 0.06、seed=7、anytime 2500 次迭代，不是保证阈值，env-rebuilt），Tube-RRT 耗时约 0.5–2.5s：

- `random_circles` square：默认障碍 0/12 可穿过，长度 11.28，瓶颈 rho 0.318；`--obstacle-radius auto`（0.023–0.058）后 12/12 可穿过，长度 11.38，瓶颈 0.476；t_shape + auto 长度 11.76。这些路径都没有骑跨障碍：小障碍绕行代价很低。
- `single_post` square：默认绕行，长度 11.18；auto（半径 0.039）仍然绕行，长度 11.11；×2 时 26 个路径节点中有 4 个骑跨柱子，长度 11.09，瓶颈 0.011。
- `post_fence` square：默认柱子（0.08 > b）失败；auto（0.039）回退成功，长度 13.54，从两柱之间整队挤过，瓶颈 0.020；同条件固定步长失败；×2 成功，4/17 个路径节点骑跨柱子，长度 10.94。线性 nearest/neighbor 和 descendant cost 更新使搜索耗时随节点数超线性增长。`formation/distance_field.py:61-68` 的纯 Python 行列扫描约 0.59-0.62s，是独立热点，当前未改。

## 验证命令与最近记录

最近验证记录：focused 16 tests，完整 63 tests；这些是最近一次验证记录，不是永久保证。

项目使用的 Conda 环境是 `../env-rebuilt`（即 `/home/eai/projects/env-rebuilt`，可用 `conda activate /home/eai/projects/env-rebuilt` 激活）。

```bash
../env-rebuilt/bin/python -m unittest tests.test_tube_rrt tests.test_maps
../env-rebuilt/bin/python -m unittest discover -s tests
MPLBACKEND=Agg ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py --formation square --seed 7
MPLBACKEND=Agg ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py --map post_fence --slot-scale 2
MPLBACKEND=Agg ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py --map single_post --slot-scale 2
MPLBACKEND=Agg ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py --map random_circles --margin-weight 0.1
MPLBACKEND=Agg ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py --map random_circles --obstacle-radius auto
MPLBACKEND=Agg ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py --map single_post --obstacle-radius 0.05 --robot-radius 0.113
# 加 --no-step-backoff 可对比旧的固定步长行为
# 结果：results/tube_rrt/README.md（索引）和各运行目录下的 run.md
```

## 环境重建

根目录包含 `environment-linux-64.yml`、`conda-linux-64.lock`、`requirements-pip-linux-64.txt` 和 `scripts/rebuild_env_linux64.sh`。

- Linux x86_64 同平台优先使用 explicit lock，它记录当前 Conda 包的精确 URL 和 build；若其中某个历史包 URL 已不可用，再使用可读 YAML 作为求解式后备。
- pip 文件只包含未由 `conda-meta` 管理的 pip-side 包；重建脚本以 `--no-deps` 安装，避免覆盖 Conda 已还原的依赖。
- 重建脚本默认创建 `../env-rebuilt`（当前使用的环境），也可通过第一个参数指定新前缀；目标已存在时会立即拒绝，绝不会删除、覆盖或修改已有环境。
- 不要复制完整环境目录。

```bash
# 首选：同平台精确重建并自动执行 pip check、核心 import 和 focused tests
scripts/rebuild_env_linux64.sh

# 指定一个尚不存在的新前缀
scripts/rebuild_env_linux64.sh /absolute/path/to/new-env

# explicit URL 失效时才使用可读 YAML 后备
conda env create --prefix ../env-rebuilt-fallback --file environment-linux-64.yml
```

## Git 交付

当前 branch 是 `curve-band-v2`，当前基础 HEAD 为 `b9e62db`，且尚未配置 remote。以下命令只显式提交本次交付文件，不包含 `openspec/`；本地发送到远程的动作是 `push`，当前尚未执行提交或推送。

```bash
# 1. 检查并提交本次交付文件
git status --short
git add \
  HANDOFF.md \
  environment-linux-64.yml \
  conda-linux-64.lock \
  requirements-pip-linux-64.txt \
  scripts/rebuild_env_linux64.sh
git commit -m "添加项目交付文档和环境重建配置"

# 2A. 尚无 origin 时添加并推送
git remote add origin <REMOTE_URL>
git push -u origin curve-band-v2

# 2B. 若 origin 已存在但地址错误，改址后推送
git remote set-url origin <REMOTE_URL>
git push -u origin curve-band-v2

# 3A. 另一台机器首次获取该分支
git clone --branch curve-band-v2 <REMOTE_URL>

# 3B. 已有 checkout、尚无本地 curve-band-v2
git fetch origin
git switch --track origin/curve-band-v2

# 3C. 已有本地 curve-band-v2，仅快进更新
git switch curve-band-v2
git pull --ff-only origin curve-band-v2
```

操作前可用 `git status --short --branch` 和 `git remote -v` 核对状态；不要覆盖远端或本地未提交改动。

## 下一步建议

保持范围聚焦。若默认体验优先，先考虑缓存或替换距离场构建；只有在更大迭代预算的实测确实需要时，再优化 search 索引，并保持现有 tie-breaking 与确定性。不要把全局曲线路径优化、portal、平滑曲线、动力学或控制器功能写成已完成。
