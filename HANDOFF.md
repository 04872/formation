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

搜索过程是 seeded SE(2) sampling（`formation/tube_rrt.py:116-122`），带 goal bias；对最近节点按上述 metric steering（`formation/tube_rrt.py:124-128`），检查新节点 clearance 和与最近节点的 tube overlap。然后在邻域内做 parent selection（`formation/tube_rrt.py:179-188`），再做 rewiring，并递归更新 descendant cost（`formation/tube_rrt.py:191-198`、`formation/tube_rrt.py:133-144`）。到目标中心足够近时，创建中心精确为目标、yaw 继承到达节点的 goal node。结果保留搜索树，并输出未平滑的 joint 折线路径及对应 radii。`progress_interval` 大于 0 时输出进度，否则默认静默。

## 可视化与计时

`scripts/visualize_tube_rrt.py` 支持：

- `--output PATH`：保存图片；未指定时交互式 `plt.show()`。
- `--formation {square,column,horizontal_line,t_shape}`。
- `--seed INT`。
- `--progress-interval INT`，非负；传 0 关闭搜索进度输出。

绘图包含精确圆障碍、树的中心投影、中心折线、每个机器人投影折线、沿路径采样的机器人圆盘，以及随姿态旋转且仅用于显示的编队框（`scripts/visualize_tube_rrt.py:62-99`）。编队框不是碰撞模型。

脚本计时输出包括 `startup_imports`、`map_build`、`tube_rrt`、`plot_build`；保存模式还输出 `save` 和 `total`，交互模式输出 `total_before_show`（`scripts/visualize_tube_rrt.py:40-44`、`scripts/visualize_tube_rrt.py:52-55`、`scripts/visualize_tube_rrt.py:102-111`）。交互窗口等待不计入绘图耗时。

默认 seed=7 square 的单次实测参考（不是保证阈值）：startup/imports 约 0.87s，map 约 0.60s，Tube-RRT 约 0.035s，plot build 约 0.73s，save 约 0.48s，total 约 2.72s；默认搜索约 78 个节点。主要耗时不是 Tube-RRT。满迭代树时线性 nearest/neighbor 和 descendant cost 更新可能变慢：clear map 约 11.76s，random circles 约 8.33s。`formation/distance_field.py:61-68` 的纯 Python 行列扫描约 0.59-0.62s，是独立热点，当前未改。

## 验证命令与最近记录

最近验证记录：focused 13 tests，完整 60 tests；这些是最近一次验证记录，不是永久保证。

```bash
../env/bin/python -m unittest tests.test_tube_rrt tests.test_maps
../env/bin/python -m unittest discover -s tests
MPLBACKEND=Agg ../env/bin/python scripts/visualize_tube_rrt.py --formation square --seed 7 --progress-interval 0 --output /tmp/tube_rrt-square.png
```

## 环境重建

根目录包含 `environment-linux-64.yml`、`conda-linux-64.lock`、`requirements-pip-linux-64.txt` 和 `scripts/rebuild_env_linux64.sh`。

- Linux x86_64 同平台优先使用 explicit lock，它记录当前 Conda 包的精确 URL 和 build；若其中某个历史包 URL 已不可用，再使用可读 YAML 作为求解式后备。
- pip 文件只包含未由 `conda-meta` 管理的 pip-side 包；重建脚本以 `--no-deps` 安装，避免覆盖 Conda 已还原的依赖。
- 重建脚本默认创建 `../env-rebuilt`，也可通过第一个参数指定新前缀；目标已存在时会立即拒绝，绝不会删除、覆盖或修改现有 `../env`。
- 不要复制完整 `env` 目录。

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
