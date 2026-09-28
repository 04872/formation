# HANDOFF

## 交接范围与当前目标

本仓库当前交付的是 rigid-formation Joint Tube-RRT 全局折线路径：把固定槽位的整队作为一个联合系统，在 SE(2) 联合构型空间中搜索一条带局部安全裕度的可行路径，并将联合路径投影为各机器人路径。当前输出仍是未平滑的折线联合路径；不包含全局曲线路径优化、portal 优化、B-spline/Bezier、动力学、控制器、MPC 或机器人闭环运动。

设计背景可对照普通 `method/编队全局路径规划.md`，但当前实现范围以源码为准。

## 关键文件图谱

- `formation/tube_rrt.py`：原实现（默认 `cell_model="orientation"`）。`TubeRRTConfig`、节点 / trace / 结果结构、`OrientationSafeCell`（yaw 切片 cell）、`EdgeCertificate`（公共 yaw witness + portal）、`TubeRRTPlanner`、`transform_slots`、`interpolate_pose`、`project_robot_paths`。相对 `HEAD` 只增加了 `cell_model` / `route_check_step` 配置、结果的 `path_nodes` / `overlap_stats` 字段和非 orientation 时的报错，搜索逻辑未改。
- `formation/tube_cell_first_order.py`：一阶近似 cell。`TubeCell`（节点位姿、`d_obs`、半径）、`PortalCertificate`（portal、相对 slack、路线长度）、`interpolate_pose`，以及 `FirstOrderCellModel`：chart 范数 `||.||_F`、批量 nearest 距离、位似 overlap、固定 yaw 截面（绘图用）。
- `formation/tube_cell_second_order.py`：二阶 cell。`SecondOrderCellModel` 继承一阶模型，加入 `beta_i phi^2` 项、射线内步长、必要条件快速拒绝、候选 portal 快速接受和 Clarabel SOCP overlap。
- `formation/tube_rrt_chart.py`：`ChartCellTubeRRTPlanner`（一阶 / 二阶 cell 共用的 RRT*：逐机器人 guarded clearance、nearest、父节点选择、rewire、认证路线与稠密 clearance 检查）、`CELL_MODELS`，以及工厂函数 `make_tube_rrt_planner`：`cell_model="orientation"` 返回原 `TubeRRTPlanner`，其余返回 `ChartCellTubeRRTPlanner`。
- `formation/map_config.py`、`formation/map_builder.py`：地图配置与构造（`random_circles`、`post_fence`、`single_post` 等）。
- `formation/__init__.py`：公开导出。
- `scripts/visualize_tube_rrt.py`：CLI、规划、绘图、计时、报告和结果索引。
- `tests/test_tube_rrt.py`：工厂分派（默认 orientation）、原 orientation 实现复现旧运行（random_circles seed 7、square×2、anytime：2143 节点、首次 236 次迭代、代价 12.406），以及 chart 范数定义、一阶位似 overlap、portal 同属两个 cell、一阶 / 二阶 cell 对真实机器人位移的界、二阶 overlap 与暴力采样一致、射线内步长、两种 cell 的确定性规划、二阶路线认证与无碰撞、rewire 后代价一致、非法 cell 名和起点碰撞。
- `tests/test_maps.py`：全部地图字段、随机圆确定性 / 边界、各类走廊和起终点验证。

## 当前算法与安全语义

联合状态 `q=(x, y, theta)`，槽位 `r_i` 在编队坐标系，机器人中心 `p_i = c + R(theta) r_i`。起点姿态为 `Pose2D(*start_xy, 0.0)`；终点中心为 `goal_xy`，yaw 继承到达节点。clearance 逐机器人计算（地图边界和每个圆障碍，再减机器人半径和安全余量），`d_obs(q)` 取所有机器人中的最小值，不使用整队外包圆或凸包膨胀。只支持 `circle` 障碍 primitive。

**三种 cell 版本**由 `TubeRRTConfig.cell_model`（脚本 `--cell`）选择，均通过 `make_tube_rrt_planner` 构造：`orientation`（默认，原实现）、`first_order`、`second_order`。三者共享 guarded clearance、采样、goal bias、步长回退、J_margin、anytime 语义，但度量、cell 与 overlap 各自独立。

**orientation cell（`cell_model="orientation"`，默认，`tube_rrt.py`）**：度量 `d_G = ||c_a - c_b|| + R_F |wrap(theta_b - theta_a)|`（`R_F` 为槽位最大半径）。每个节点的 `OrientationSafeCell` 以节点中心为圆心，在 `yaw_slices`（默认 16）个 yaw 上计算 guarded clearance，用 `R_F`-Lipschitz 解析下包络得到各 yaw 下的安全圆盘半径（乘 `cell_eta` 收缩），并取包含节点 yaw 的正半径 yaw 区间。相邻 cell 的 overlap 用 `EdgeCertificate`：找一个两侧都有效的公共 yaw witness，在该 yaw 下两圆盘严格相交，portal 取交集中的点；认证路线为“节点 → 旋转到 witness yaw → 平移到 portal → 平移到下一节点 → 旋转到下一节点 yaw”。`result.path_poses` 是展开后的这条路线，`bottleneck` 为路线上的最小局部 guarded clearance。

**局部 chart（一阶 / 二阶 cell）**：对节点 `q_k`，`u = q.xy - q_k.xy`、`phi = wrap(theta - theta_k)`、`a_i = R(theta_k) r_i`、`J` 为 90° 旋转。真实位移 `Δp_i = u + (R(phi) - I) a_i`，一阶近似为 `u + J a_i phi`。chart 范数 `||q - q_k||_F = max_i ||u + J a_i phi||`（等于机体坐标下的 `max_i ||R(-theta_k) u + J r_i phi||`），用于 nearest、邻域、steer 步长和路线长度。

**一阶 cell（`cell_model="first_order"`，`tube_cell_first_order.py`）**：`U_k = {q : ||q - q_k||_F < rho_k}`，`rho_k = eta d_obs(q_k)`。不同节点的 chart 不同（`a_i` 随 `theta_k` 旋转），所以 overlap 用沿 chart 线段 `q_a -> q_b` 的判据 `rho_a/n_a + rho_b/n_b > 1`（`n_a = ||q_b - q_a||_{F,a}`、`n_b = ||q_a - q_b||_{F,b}`）；yaw 相同时两范数相等，正好是 `||q_a - q_b||_F < rho_a + rho_b`，其他情况是充分条件。portal 取线段上按两侧 slack 相等的点。一阶 cell 忽略了二阶余项，**不是**碰撞证书：`eta = 1`、square、`rho = 0.4` 时，cell 内真实机器人位移最多可到 `1.04 rho`（见测试）。

**二阶 cell（`cell_model="second_order"`，`tube_cell_second_order.py`）**：`U_k = {q : max_i (||u + J a_i phi|| + beta_i phi^2) < d_k}`，`beta_i = ||r_i||/2`，`d_k = eta d_obs(q_k)`。因为 `||(R(phi) - I) r_i - J r_i phi|| <= ||r_i|| phi^2 / 2`，cell 内每个机器人的真实位移都小于 `d_k <= d_obs(q_k)`，所以 cell 严格无碰撞。overlap 判定两个 cell 是否有公共构型：先用两个必要条件快速拒绝（`|δ|` 超过两侧 yaw 范围之和；某个机器人在 `q_a`、`q_b` 的真实位置相距 `>= d_a + d_b`），再在 chart 线段上的候选点（两端节点、t = 0.5 / 0.25 / 0.75）快速接受，剩下的才解 3D 主变量的小 SOCP（`max s`，约束为 `||u + J a_i phi|| + beta_i z <= d(1 - s)`，`z >= phi^2` 用旋转二阶锥表示），`s* > 0` 即 overlap，最优点为 portal，并用精确 slack 复核。

一阶 / 二阶 cell 共有：半径上限 `pi * spread / 2`（`spread` 为最大槽位间距），保证 cell 的 yaw 范围在 `(-pi, pi)` chart 内、在 chart 中是凸集。相邻节点的认证路线是 `q_a -> portal -> q_b`，两段各位于一个凸 cell 内；`result.path_poses` 就是这条路线（节点与 portal 交替），`path_nodes` 为树节点下标，`bottleneck` 为沿路线按 `route_check_step`（0.02）稠密采样的最小 guarded clearance（负值即碰撞）。边代价为 portal 路线长度 `||q_p - q_a||_{F,a} + ||q_b - q_p||_{F,b}`，`margin_weight = w > 0` 时乘 `1 + w / min(radius_a, radius_b)`（J_margin）。

**搜索（一阶 / 二阶）**：seeded SE(2) 采样（goal bias 0.12），按 `||.||_F` 找最近节点并 steer（`metric_step` 0.45）；步长回退时每次减半，下限为 `0.9 * inner_step`：一阶取 `rho_near`，二阶取射线上 `s + beta_max (c s)^2 < d` 的解（`c` 为单位 F 长度的转角），保证新节点落在最近 cell 内部，所以最后一次尝试必然 overlap。之后按“邻居代价 + 距离”升序做父节点选择（下界不优即停止），再 rewire 并更新后代代价。`stop_on_first_goal=True`（库默认）时首次到达即返回；`False` 为 anytime：跑满迭代，只保留更优的 goal 节点。`result.overlap_stats` 记录 overlap 调用、快速拒绝 / 接受、SOCP 次数和 SOCP 接受数。

## 可视化与计时

`scripts/visualize_tube_rrt.py` 支持：

- `--out-dir DIR`：结果根目录，默认 `results/tube_rrt`（`results/` 已被 gitignore）。
- `--map {random_circles,post_fence,single_post}`，默认 `random_circles`。
- `--formation square`：当前只做 square 队形（其他队形暂不支持）；`--slot-scale K` 把槽位整体放大 K 倍（默认 1）。
- `--robot-radius R`：机器人外接圆半径，默认 0.113 m（TurtleBot3 Burger 底盘 138×178 mm 的外接圆半径）；同时决定默认槽位间距（`FormationLibrary.build_default(R)`）。
- `--safety-margin M`：障碍膨胀余量，默认 0.06 m。
- `--obstacle-radius {R | MIN:MAX | auto}`：障碍半径。`random_circles` 取 R 或区间 [MIN, MAX]；柱子地图只接受单值。`auto` 按“可穿过上限” b 自动取值：`random_circles` 取 (0.3b, 0.8b)，柱子取 0.5b；b ≤ 0 时报错。不传时保持地图默认值。
- `--obstacle-count N`：`random_circles` 的障碍个数。
- `--seed INT`：规划器 seed，同时是 `random_circles` 的地图 seed。
- `--iterations INT`：迭代预算，默认 2500。
- `--anytime`：跑满迭代预算并保留代价最小的目标路径；不加时首次连到目标即停止。
- `--cell {orientation,first_order,second_order}`：cell 版本，默认 `orientation`（原实现，结果目录名与以前相同）；`first_order` 为一阶近似 cell，`second_order` 为二阶、严格无碰撞、SOCP overlap。
- `--cell-shrink E`：cell 收缩系数 eta，默认 0.98，三种版本通用（orientation 乘在圆盘半径上；一阶 `rho_k`、二阶 `d_k` = `E * d_obs(q_k)`）。
- `--yaw-slices N`：orientation cell 的 yaw 切片数，默认 16（只对 orientation 生效）。
- `--margin-weight W`：J_margin 权重，默认 0。
- `--no-step-backoff`：关闭步长回退（旧的固定步长行为）。
- `--progress-interval INT`，非负，默认 500；传 0 关闭搜索进度输出。
- `--show`：保存后再交互式显示全部图。

**测试地图与几何前提**：可穿过障碍半径上限 `b = max(凸包边界上相邻机器人间距的一半) − robot_radius − safety_margin`，半径小于 b 的障碍才可能从机器人之间穿过。脚本在控制台 `geometry` 行、`run.md` 和索引中给出 b 与可穿过障碍个数；个数为 0 时打印提示。Burger 半径 0.113、margin 0.06 下：

| 编队 | b（×1） | b（×2） |
| --- | --- | --- |
| square | 0.077 | 0.327 |

`random_circles` 默认障碍半径 0.12–0.28、`single_post` 默认 0.15、`post_fence` 默认 0.08，都大于 ×1 下的 0.077，所以默认场景只能绕行（这是正确行为）。要展示障碍从机器人之间穿过，可以用 `--obstacle-radius auto` 缩小障碍，或者用 `--slot-scale 2` 放大编队：

- `post_fence`（`PostFenceConfig`）：x=0 处一排柱子，间距 1.1 m，半径 0.08 m。配合 `--slot-scale 2` 时整队无法从两柱之间通过，只能让柱子从机器人之间穿过。
- `single_post`（`SinglePostConfig`）：起终点连线上只有一根半径 0.15 m 的柱子。`--slot-scale 2` 时最短路径骑跨柱子。

每次运行保存到 `<out-dir>/<map>_seed<seed>/<变体>/`，变体名依次由以下部分组成：编队名；`cell1`（一阶）或 `cell2`（二阶），orientation 不加标记；`x<K>`；`r<R>`、`sm<M>`（非默认机器人半径 / 余量）；`obs<spec>`、`n<N>`（障碍半径 / 个数）；`anytime_it<N>` 或 `first_goal`；`wm<W>`；`eta<E>`（非默认 eta）；`fixedstep`。同参数重跑会覆盖同一目录（搜索是确定性的）。目录内容：

- `run.md`：给人阅读的运行报告，含复现命令、代码版本、结果与配置表、耗时、全部图和完整控制台输出。
- `overview.png`、`1_tree.png`、`2_growth.png`、`3_tube.png`、`4_formation.png`、`5_formation_frames.png`、`6_convergence.png`。
- `path.csv`：认证路线航点（节点与 portal 交替，cx, cy, theta, guarded clearance）和各机器人投影坐标；`summary.json`：机器可读摘要。

每次运行后自动重新生成 `<out-dir>/README.md`，汇总所有运行（cell 版本、节点数、首次/最终代价、路线长度、路线最小 clearance、障碍夹在机器人之间的航点数、SOCP 次数、耗时），并链接到各自的 `run.md`。“障碍夹在机器人之间”指障碍中心落在机器人中心的凸包内；树图用粉色圈标出这类节点，编队图用粉色填充这类姿态。

脚本以 `TubeRRTConfig(record_trace=True)` 规划，`TubeRRTResult.trace` 逐迭代记录采样、最近节点、steer 结果、cell 半径、状态（`collision`/`no_overlap`/`added`/`goal`）和 rewire；默认关闭，不影响搜索与随机序列。各图含义：

- `tree`：最终树的 (x, y) 投影，边和节点按插入顺序着色，节点短线表示 yaw；叠加被拒绝的 steer 点（碰撞 / 不重叠）和被 rewire 移除的旧边。节点多于 600 时不画 yaw 短线，rewire 多于 300 时只在图例中给出数量。
- `growth`：由 trace 回放的 6 个快照（首次到达目标前 3 张、首次到达、之后 1 张、最终），标出当次 sample、nearest、q_new 及其 cell 在节点 yaw 处的 xy 截面（`phi = 0` 时一阶、二阶都是半径为 cell 半径的圆盘）和当时的最优路径。
- `convergence`：最优 goal 代价随迭代的阶梯下降曲线，以及接受节点 / 碰撞拒绝 / 不重叠拒绝 / rewire 的累计数。
- `tube`：orientation 版本与原来相同（展开路线上的固定 yaw 安全圆盘、yaw 包络曲面、沿路线的局部 clearance、各树边的认证路线长度）。一阶 / 二阶版本：左上为路径 cell 的节点 yaw 截面与 portal；右上为 (x, y, theta) 中的 cell 形状（各 yaw 截面 `D_k(theta) = ∩_i B(q_k.xy - J a_i phi, radius - beta_i phi^2)` 叠成的曲面）；左下为沿认证路线稠密采样的 guarded clearance（负值涂红）；右下为 slack 最小的 portal 处两个 cell 的固定 yaw 截面（虚线为各机器人圆盘，实线为交集）和 portal 姿态。
- `formation`：视野裁剪到路径附近。沿联合路径按 `平移 + R_F·转角` 等间距采样 10 个姿态（在 chart 线段上插值，不局限于航点），每个姿态一种颜色（turbo 色表）并在中心编号；轮廓线连接各机器人，粉色填充表示障碍夹在机器人之间。机器人轨迹按每条边稠密插值绘制，瓶颈节点处画出半径 `r + margin + rho` 的圆（与最近障碍相切）；下方为 theta 沿路径曲线。
- `formation_frames`：6 个关键帧，取障碍夹在机器人之间的航点区间（没有时取 clearance 最小的航点附近）。每个子图以“上一节点 + 当前节点”为中心、比例尺相同；当前姿态为蓝色轮廓，上一节点为灰色虚线，黑箭头为中心位移，被夹住的障碍描粉色边；标题给出 theta、guarded clearance 和这一步的 `||.||_F`。
- 总览：tree、cell 截面、formation、clearance 曲线四宫格。

脚本计时输出包括 `startup_imports`、`map_build`、`tube_rrt`、`plot_build`、`save`、`total`，同时写入 `run.md` 和 `summary.json`。

orientation 版本的旧命令 `--map random_circles --seed 7 --slot-scale 2 --anytime` 在当前代码下 `path.csv` 与已保存结果逐字节一致（rewire 计数 972 vs 已存 970，来自旧环境 `../env`；纯 `HEAD` 代码同样是 972）。

当前只测试 `--slot-scale 2 --anytime` 场景（square、Burger 半径 0.113、margin 0.06、seed 7、2500 次迭代、eta 0.98；单次实测，不是保证阈值，env-rebuilt）。`--obstacle-radius auto` 代码逻辑保留，但不在测试范围内。三种 cell 的 Tube-RRT 耗时：orientation 约 0.49–0.64s，一阶约 0.32s，二阶约 0.50s（SOCP 约 700–800 次，每次约 0.1–0.2 ms；约 50% 的 overlap 被必要条件快速拒绝，约 45% 被候选 portal 快速接受）。

| 地图 | orientation：代价 / 最小 clearance | 一阶：代价 / 最小 clearance | 二阶：代价 / 最小 clearance | 夹障碍路径节点（orientation / 一阶 / 二阶） |
| --- | --- | --- | --- | --- |
| `random_circles` | 12.406 / 0.150 | 12.040 / 0.218 | 12.031 / 0.218 | 0 / 0 / 0 |
| `single_post` | 11.089 / 0.012 | 11.993 / 0.179 | 11.996 / 0.179 | 14 / 0 / 0 |
| `post_fence` | 11.363 / 0.124 | 10.806 / 0.065 | 10.802 / 0.065 | 12 / 9 / 9 |

orientation 的 clearance 是展开路线上的局部 guarded clearance，一阶 / 二阶是认证路线上稠密采样的 guarded clearance，两者口径不同。对全部树边稠密检查，这些场景中一阶 cell 也没有产生碰撞边；但一阶 cell 内确实存在真实位移超过 `rho` 的构型，只有二阶 cell 是证书。`formation/distance_field.py:61-68` 的纯 Python 行列扫描约 0.6s，是独立热点，当前未改。

## 验证命令与最近记录

最近验证记录：focused 20 tests（`test_tube_rrt` 12 + `test_maps` 8），完整 67 tests；这些是最近一次验证记录，不是永久保证。

项目使用的 Conda 环境是 `../env-rebuilt`（即 `/home/eai/projects/env-rebuilt`，可用 `conda activate /home/eai/projects/env-rebuilt` 激活）。

```bash
../env-rebuilt/bin/python -m unittest tests.test_tube_rrt tests.test_maps
../env-rebuilt/bin/python -m unittest discover -s tests
for map in random_circles single_post post_fence; do
  for cell in orientation first_order second_order; do
    MPLBACKEND=Agg ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py --map $map --slot-scale 2 --anytime --cell $cell
  done
done
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
