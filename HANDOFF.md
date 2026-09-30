# HANDOFF

## 交接范围与当前目标

本仓库当前交付的是 rigid-formation Joint Tube-RRT 全局折线路径：把固定槽位的整队作为一个联合系统，在 SE(2) 联合构型空间中搜索一条带局部安全裕度的可行路径，并将联合路径投影为各机器人路径。当前输出仍是未平滑的折线联合路径；不包含全局曲线路径优化、portal 优化、B-spline/Bezier、动力学、控制器、MPC 或机器人闭环运动。

设计背景可对照普通 `method/编队全局路径规划.md`，但当前实现范围以源码为准。

## 关键文件图谱

- `formation/tube_rrt.py`：原实现（默认 `cell_model="orientation"`）。`TubeRRTConfig`、节点 / trace / 结果结构、`OrientationSafeCell`（yaw 切片 cell）、`EdgeCertificate`（公共 yaw witness + portal）、`TubeRRTPlanner`、`transform_slots`、`interpolate_pose`、`project_robot_paths`。相对 `HEAD` 只增加了 `cell_model` / `route_check_step` 配置、结果的 `path_nodes` / `overlap_stats` 字段和非 orientation 时的报错，搜索逻辑未改。
- `formation/tube_cell_first_order.py`：一阶近似 cell。`TubeCell`（节点位姿、`d_obs`、半径）、`PortalCertificate`（portal、相对 slack、路线长度）、`interpolate_pose`，以及 `FirstOrderCellModel`：chart 范数 `||.||_F`、批量 nearest 距离、位似 overlap、固定 yaw 截面（绘图用）。
- `formation/tube_cell_second_order.py`：二阶 cell。`SecondOrderCellModel` 继承一阶模型，加入 `beta_i phi^2` 项、射线内步长、必要条件快速拒绝、候选 portal 快速接受和 Clarabel SOCP overlap。
- `formation/tube_rrt_chart.py`：`ChartCellTubeRRTPlanner`（一阶 / 二阶 cell 共用的 RRT*：逐机器人 guarded clearance、nearest、父节点选择、rewire、认证路线与稠密 clearance 检查）、`CELL_MODELS`，以及工厂函数 `make_tube_rrt_planner`：`cell_model="orientation"` 返回原 `TubeRRTPlanner`，`"polyhedral"` 返回 `PolyhedralFrontierPlanner`（可选参数 `frontier_config`），其余返回 `ChartCellTubeRRTPlanner`。
- `formation/polyhedral_cell.py`：polyhedral cell。`PolyhedralCell`（seed 位姿、边界行 `normals / offsets` 与对应 `pairs`、guard 半径 `guard`、由约束决定的 yaw 半宽 `yaw_limit`、S_0 内切圆 `inradius / center_offset`、`broadphase_pairs`、Halton 内部样本与体积估计）和 `PolyhedralCellModel`：proximity query（4 面墙 + 圆障碍）、两级 active pair、建 cell、成员判定 / slack、`translational_extent`（ℓ_i(u, Δθ)）、`ray_extent`、`boundary_radii`（三维边界曲面）、`slice_polygon`（固定 yaw 截面及每条边所属约束，−1 = guard 弧）、overlap（快速拒绝 / 线段候选快速接受 / 方向部分 LP / 含 guard 锥的 SOCP）。`stats` 含 `pose_queries`、`pair_queries` 等查询计数。
- `formation/tube_rrt_frontier.py`：`FrontierConfig`（含 p_region 调度 `region_schedule` = switch / constant / exp）与 `PolyhedralFrontierPlanner`（标准 RRT* 主体 + region 采样通道、每次迭代至多 1 个 cell、父节点选择 / rewire、认证路线、T_first / N_query 统计）。`frontier_snapshot()` 和 `expansion_log` 供绘图使用。
- `scripts/ablate_region_sampling.py`：p_region 消融（常数 0 / 0.2 / … / 1.0、switch、exp；三张地图 × 多个 planner seed，并行），输出 `results/region_ablation/{runs.csv,summary.csv,README.md,ablation.png,convergence.png}`。
- `scripts/compare_region_schemes.py`：region 采样**方案**对比（uniform、v1 `f36dc0c` 多 seed 预建、v2 `bededcf` 统一外法向、v3 `ffe0d5b` 按边界来源、v4 facet 几何；各自默认设置与统一 p=0.85），每个方案在 `/tmp/formation_schemes/<commit>` 的 detached worktree 中运行，T_first / N_query 在规划器外统一测量；输出 `results/region_schemes/{runs.csv,summary.csv,README.md,comparison.png}`，`--plot-only` 只重画。`scripts/summarize_region_ablation.py`：单方案 p_region 消融的一页汇总图 `results/region_ablation/overview.png`。
- `formation/map_config.py`、`formation/map_builder.py`：地图配置与构造（`random_circles`、`post_fence`、`single_post` 等）。
- `formation/__init__.py`：公开导出。
- `scripts/visualize_tube_rrt.py`：CLI、规划、绘图、计时、报告和结果索引。
- `tests/test_tube_rrt.py`：工厂分派（默认 orientation）、原 orientation 实现复现旧运行（random_circles seed 7、square×2、anytime：2143 节点、首次 236 次迭代、代价 12.406），以及 chart 范数定义、一阶位似 overlap、portal 同属两个 cell、一阶 / 二阶 cell 对真实机器人位移的界、二阶 overlap 与暴力采样一致、射线内步长、两种 cell 的确定性规划、二阶路线认证与无碰撞、rewire 后代价一致、非法 cell 名和起点碰撞。
- `tests/test_maps.py`：全部地图字段、随机圆确定性 / 边界、各类走廊和起终点验证。
- `tests/test_polyhedral_frontier.py`（19 个；另检查 `frontier_pieces` 恰在截面边界（facet 行松弛为 0、guard 弧半径 = r_g）；live 扩展方向与 facet 法向 / guard 外法向夹角 < 90°、Δθ ∈ {0, ±s} 且使旋转裕度最大、q_new 在源 cell 外且裕度 > 0、live q_new 不被任何 cell 覆盖、每个 cell 的方向数有界；region 节点离开源 cell、旋转裕度 > 0、距源 cell 中心 ≤ r_g + outer_reach）：公共旋转半径 ρ、边界行来自 broadphase 且 guard 来自其余 pair、剪枝后的 cell 是全部 broadphase 行 cell 的子集且几乎相同、cell 内任意构型真实 clearance ≥ d_s（蒙特卡洛）、yaw 半宽 = min(π/2, R_in/ρ) 且超出后截面为空、ℓ 恰好落在边界、射线步长、三维边界曲面、截面多边形与成员判定一致、portal 严格在两 cell 内且与暴力采样一致（LP / SOCP 均被调用）、p_region 三种调度、每次迭代至多 1 个 cell（`cells_built = 1 + 迭代 − no_progress`）与认证路线、p_region = 0 / 1 两个极端、post_fence 可达、起点碰撞。

## 当前算法与安全语义

联合状态 `q=(x, y, theta)`，槽位 `r_i` 在编队坐标系，机器人中心 `p_i = c + R(theta) r_i`。起点姿态为 `Pose2D(*start_xy, 0.0)`；终点中心为 `goal_xy`，yaw 继承到达节点。clearance 逐机器人计算（地图边界和每个圆障碍，再减机器人半径和安全余量），`d_obs(q)` 取所有机器人中的最小值，不使用整队外包圆或凸包膨胀。只支持 `circle` 障碍 primitive。

**三种 cell 版本**由 `TubeRRTConfig.cell_model`（脚本 `--cell`）选择，均通过 `make_tube_rrt_planner` 构造：`orientation`（默认，原实现）、`first_order`、`second_order`。三者共享 guarded clearance、采样、goal bias、步长回退、J_margin、anytime 语义，但度量、cell 与 overlap 各自独立。

**orientation cell（`cell_model="orientation"`，默认，`tube_rrt.py`）**：度量 `d_G = ||c_a - c_b|| + R_F |wrap(theta_b - theta_a)|`（`R_F` 为槽位最大半径）。每个节点的 `OrientationSafeCell` 以节点中心为圆心，在 `yaw_slices`（默认 16）个 yaw 上计算 guarded clearance，用 `R_F`-Lipschitz 解析下包络得到各 yaw 下的安全圆盘半径（乘 `cell_eta` 收缩），并取包含节点 yaw 的正半径 yaw 区间。相邻 cell 的 overlap 用 `EdgeCertificate`：找一个两侧都有效的公共 yaw witness，在该 yaw 下两圆盘严格相交，portal 取交集中的点；认证路线为“节点 → 旋转到 witness yaw → 平移到 portal → 平移到下一节点 → 旋转到下一节点 yaw”。`result.path_poses` 是展开后的这条路线，`bottleneck` 为路线上的最小局部 guarded clearance。

**局部 chart（一阶 / 二阶 cell）**：对节点 `q_k`，`u = q.xy - q_k.xy`、`phi = wrap(theta - theta_k)`、`a_i = R(theta_k) r_i`、`J` 为 90° 旋转。真实位移 `Δp_i = u + (R(phi) - I) a_i`，一阶近似为 `u + J a_i phi`。chart 范数 `||q - q_k||_F = max_i ||u + J a_i phi||`（等于机体坐标下的 `max_i ||R(-theta_k) u + J r_i phi||`），用于 nearest、邻域、steer 步长和路线长度。

**一阶 cell（`cell_model="first_order"`，`tube_cell_first_order.py`）**：`U_k = {q : ||q - q_k||_F < rho_k}`，`rho_k = eta d_obs(q_k)`。不同节点的 chart 不同（`a_i` 随 `theta_k` 旋转），所以 overlap 用沿 chart 线段 `q_a -> q_b` 的判据 `rho_a/n_a + rho_b/n_b > 1`（`n_a = ||q_b - q_a||_{F,a}`、`n_b = ||q_a - q_b||_{F,b}`）；yaw 相同时两范数相等，正好是 `||q_a - q_b||_F < rho_a + rho_b`，其他情况是充分条件。portal 取线段上按两侧 slack 相等的点。一阶 cell 忽略了二阶余项，**不是**碰撞证书：`eta = 1`、square、`rho = 0.4` 时，cell 内真实机器人位移最多可到 `1.04 rho`（见测试）。

**二阶 cell（`cell_model="second_order"`，`tube_cell_second_order.py`）**：`U_k = {q : max_i (||u + J a_i phi|| + beta_i phi^2) < d_k}`，`beta_i = ||r_i||/2`，`d_k = eta d_obs(q_k)`。因为 `||(R(phi) - I) r_i - J r_i phi|| <= ||r_i|| phi^2 / 2`，cell 内每个机器人的真实位移都小于 `d_k <= d_obs(q_k)`，所以 cell 严格无碰撞。overlap 判定两个 cell 是否有公共构型：先用两个必要条件快速拒绝（`|δ|` 超过两侧 yaw 范围之和；某个机器人在 `q_a`、`q_b` 的真实位置相距 `>= d_a + d_b`），再在 chart 线段上的候选点（两端节点、t = 0.5 / 0.25 / 0.75）快速接受，剩下的才解 3D 主变量的小 SOCP（`max s`，约束为 `||u + J a_i phi|| + beta_i z <= d(1 - s)`，`z >= phi^2` 用旋转二阶锥表示），`s* > 0` 即 overlap，最优点为 portal，并用精确 slack 复核。

一阶 / 二阶 cell 共有：半径上限 `pi * spread / 2`（`spread` 为最大槽位间距），保证 cell 的 yaw 范围在 `(-pi, pi)` chart 内、在 chart 中是凸集。相邻节点的认证路线是 `q_a -> portal -> q_b`，两段各位于一个凸 cell 内；`result.path_poses` 就是这条路线（节点与 portal 交替），`path_nodes` 为树节点下标，`bottleneck` 为沿路线按 `route_check_step`（0.02）稠密采样的最小 guarded clearance（负值即碰撞）。边代价为 portal 路线长度 `||q_p - q_a||_{F,a} + ||q_b - q_p||_{F,b}`，`margin_weight = w > 0` 时乘 `1 + w / min(radius_a, radius_b)`（J_margin）。

**搜索（一阶 / 二阶）**：seeded SE(2) 采样（goal bias 0.12），按 `||.||_F` 找最近节点并 steer（`metric_step` 0.45）；步长回退时每次减半，下限为 `0.9 * inner_step`：一阶取 `rho_near`，二阶取射线上 `s + beta_max (c s)^2 < d` 的解（`c` 为单位 F 长度的转角），保证新节点落在最近 cell 内部，所以最后一次尝试必然 overlap。之后按“邻居代价 + 距离”升序做父节点选择（下界不优即停止），再 rewire 并更新后代代价。`stop_on_first_goal=True`（库默认）时首次到达即返回；`False` 为 anytime：跑满迭代，只保留更优的 goal 节点。`result.overlap_stats` 记录 overlap 调用、快速拒绝 / 接受、SOCP 次数和 SOCP 接受数。

**polyhedral cell + frontier 搜索（`cell_model="polyhedral"`，`polyhedral_cell.py` / `tube_rrt_frontier.py`，分支 `polyhedral-frontier`）**：要求所有槽位到编队中心距离相同 `||s_i|| = ρ`（square × 2 即相邻机器人 1 m、ρ = 0.707 m；实现取 `ρ = max ||s_i||`，对不等距编队也保守）。对 seed `q_0` 做一次位姿 proximity query（逐机器人对障碍 component：4 面墙 + 每个圆）得 guarded clearance `d_ij` 与分离方向 `n_ij`，约束 `n_ij^T Δc − ρ|Δθ| ≥ −(d_ij − d_s)`（`d_s` = `safety_distance`，默认 0.02 m，另加在机器人半径和安全余量之上）。

- **两级 active pair**：一级 broadphase 取 `d_ij < active_range`（1.0 m）的 pair；二级只保留真正构成 `S_0 = P(0)` 边界的行，不设每机器人固定上限：按 offset 升序，法向近平行（`n_a·n_b > cos ε`，ε = 5°）的只保留更紧者，并把它的 offset 降到 `min(e_a, e_b − ||n_a − n_b|| r_g)`，使它在 guard 圆内仍蕴含被丢弃的行（完全平行时不变）；再用 guard 圆的外接 32 边形裁剪，丢掉在 guard 内冗余的行。因为所有行和 guard 的 `ρ|Δθ|` 系数都是 1，`P(Δθ) = S_0 ⊖ B(ρ|Δθ|)`，所以在 Δθ = 0 判定的冗余对所有 yaw 都成立（剪枝精确；测试验证剪枝后 cell ⊆ 全部 broadphase 行的 cell，且差异 < 1%）。
- **guard 是廉价谓词** `g(q) = ||Δc|| + ρ|Δθ| − r_g ≤ 0`，`r_g = min(d_inactive − d_s, max_extent)`，`d_inactive` 为 broadphase 之外 pair 的最小 clearance；不再展开成 12 个半空间。`C = {q ∈ C_dir : g(q) ≤ 0}`。
- **yaw 区间由约束决定**：π/2 只是 chart 范围。用 3 变量 SOCP 求 `S_0` 的内切圆（`n^T x + e ≥ r`，`||x|| + r ≤ r_g`），`yaw_limit = min(π/2, R_in/ρ)`，超出后截面为空（测试验证）；实测大量 cell 的 yaw 半宽远小于 π/2。cell 是 `S_0` 上的“双锥”。因为每个机器人位移 ≤ `||Δc|| + ρ|Δθ|`，cell 内任意构型的真实 clearance ≥ d_s（严格证书，测试用蒙特卡洛验证）。`radius` = min(最小 offset, r_g)。

cell overlap：yaw 区间或 guard 圆不相交时快速拒绝；节点连线上 5 个候选点有严格正 slack 则快速接受；否则先只解方向部分的 LP（两组行 + yaw 界，guard 仅以外包盒放松），无解即拒绝；有候选且 portal 满足两个 guard 即接受；否则再解加入两个 guard 二阶锥的 SOCP（Clarabel），最优点为 portal，精确 slack 复核。两 cell 都凸，路线 `q_a → q_p → q_b` 每段位于一个 cell 内，因此认证路线处处 clearance ≥ d_s。度量 / 边长为 `d_G = ||Δc|| + ρ|Δθ|`。

搜索（标准 RRT* 主体 + 有界比例的 region 采样通道）：每次迭代抽 `ξ ~ U(0, 1)`，`ξ < p_region` 走 region 通道，否则走 uniform 通道。`p_region` 调度：`switch`（默认，首解前 `region_before` = 0.5，之后 `region_after` = 0.2）、`constant`（`region_probability`）、`exp`（`p_min + (p_max − p_min) e^{−k t}`）。

- **frontier 直接来自 cell 截面几何**（`PolyhedralCellModel.frontier_pieces`）：节点 yaw 截面 S_0 的多边形边界分成若干片，每片按来源给出常数个扩展方向，不再枚举 (u, Δθ) 扇形。
  - **obstacle facet**：active 障碍行 `n^T Δc + e = 0` 的一条边。障碍行是凸障碍的支撑半空间，对该 pair 处处成立，所以不沿违反约束的外法向 `−n` 推进，而是沿切向 `±t` 绕行：两个方向 `normalize(w_t (±t) + w_g u_goal_safe + w_n n)`，其中 `u_goal_safe` 是去掉指向障碍分量后的目标方向，`n` 为不违反约束的法向分量（`obstacle_weights` = (w_t, w_g, w_n) = (1, 0.3, 0.2)）。
  - **guard 弧**：validity guard 起作用的圆弧，按 `max_arc`（π/4）切块，每块一个方向 `normalize(w_t u_tan + w_g u_goal + w_n u_out)`，`u_out` 为径向外法向，切向取朝目标的一侧（`guard_weights` = (0.3, 0.5, 1)）。
  - 每个 cell 的扩展数 ≤ guard 块数 + 2 × 边界行数（实测平均约 7 个）。
  - **代表点与 q_new**：代表点 b 取 facet 中点 / guard 块中点角处的弧点；沿 u 解析求离开 S_0 的距离 `t_exit`（对 guard 圆与各行），`q_new = b + (t_exit + sample_offset) u`（δ = 0.1 m），即刚跨出当前 cell。
  - **Δθ ∈ {+s, 0, −s}**（`yaw_step` s = 0.15 rad）：取使 outer 行的**精确旋转裕度** `min_k n_k^T (Δc + (R(Δθ) − I) R_0 s_{i_k}) + e_k` 最大者（平局取 0）。与 `−ρ|Δθ|` 不同，它区分旋转方向，能选出让夹住障碍的机器人离开障碍的整体转向。
  - **outer rows**：cell 另存 `outer_normals / outer_offsets / outer_robots`，即 `d − d_s < r_g + outer_reach`（0.5 m）的全部 pair 的未剪枝行，不进入证书，只用于算旋转裕度和校验 q_new。
  - **q_new 预先校验**（在建 frontier 时一次完成，确定性，无重试）：旋转裕度 > 0（`frontier_reject_obstacle`）、`d_G(c_0, q) ≤ r_g + outer_reach`（`far`）、在源 cell 外（`inside`）、在地图内（`bounds`）、不被其他 cell 覆盖（`covered`）。通过的保留为 live 扩展；后续 cell 覆盖其 q_new 时失效（`frontier_covered`）。
- **region 通道**：
  - 从 live 扩展中按 `exp(S/τ)` 抽一个。分数 `S = α min(ℓ/ℓ_ref, 1) + β U + γ (G+1)/2`，ℓ 为 frontier 片长度，`U` 为 q_new 之后沿 u 的探针中不在 union 内的比例，G 为 b → q_new 的目标进度；不做距离查询。
  - 被采用后若新 cell 被拒（redundant / 无 overlap 等），该扩展直接丢弃（`frontier_dropped`）；没有 live 扩展时回退 uniform（`region_fallback`）。
  - 在 `q_new` 重新做 proximity query 建 cell；要求 `ρ_new ≥ min_new_ratio`，并与源 cell（不重叠时退而找其他邻居）有 portal 证书。
- **uniform 通道**：`q_rand ~ U(SE(2))`（goal bias 0.12），d_G 最近节点（不含 goal 节点），同样 steer 且步长 ≤ `metric_step`。
- 两个通道都**只对 `q_new` 建 1 个 cell**（一次位姿 proximity query），不预建候选 cell；region 样本要求 `ρ_new ≥ min_new_ratio`（0.05，否则 `redundant`），uniform steer 步长过小记为 `no_progress`。随后与 parent 做 overlap 认证，在“可能重叠”（guard 圆 / yaw 区间相交）的至多 12 个节点中选父节点并 rewire，更新 frontier 覆盖。
- 目标：目标位姿（goal_xy, θ_new）落在 C_new 内即加 goal 节点（共享 C_new，不参与父节点选择 / rewire / nearest）。anytime 语义与 chart 版本相同。
- `result.overlap_stats`：cell / 查询计数（`pose_queries` = 构造 cell 数 = N_query，`pair_queries` = robot-obstacle 距离对数）、overlap / LP / SOCP 计数、region / uniform 迭代数与接受节点数、frontier 统计、各类拒绝，以及 `first_goal_time_s`（T_first）、`first_goal_cost`（C_first）、`first_goal_pose_queries`、`plan_time_s`。

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
- `--cell polyhedral` 时另有：`--region-schedule {switch,constant,exp}`（默认 switch）、`--region-prob P`（constant，变体名加 `p<P>`）、`--region-before / --region-after`（switch，默认 0.5 / 0.2，非默认时加 `p<before>-<after>`）、`--region-max / --region-min / --region-decay`（exp，加 `pexp<max>-<min>-k<k>`）、`--safety-distance D_S`（0.02）、`--active-range R`（1.0）、`--score-weights A B G`（1 1 1）、`--guard-weights W_T W_G W_N`（0.3 0.5 1）、`--obstacle-weights W_T W_G W_N`（1 0.3 0.2）、`--yaw-step S`（0.15）、`--sample-offset DELTA`（0.1）。变体名标记为 `cellP`；多出 `7_frontier.png`（全部 cell 的节点 yaw 截面构成的 union；已采用的 region 扩展 `b → q_new`（蓝 = guard 弧沿外法向，橙 = obstacle facet 沿切向）、仍 live 的扩展（点线）及其 Δθ 符号（▲ / ● / ▼）；ρ_new / ρ_overlap 直方图、region / uniform 接受节点累计数与 p_region 调度曲线）和 `8_tube_3d.png`（路径 cell 在 (x, y, θ) 中的三维实体，两个视角，底面为障碍投影）；`3_tube.png` 右上也画三维实体（`boundary_radii` 给出的半透明边界曲面，yaw chart 截断处加平顶），右下为最紧 portal 处的截面（实线边 = active 障碍平面，虚线边 = guard 弧）；`run.md` 额外列出 T_first / C_first / N_query、LP / SOCP、拒绝、采样通道和 frontier 统计以及 `frontier.*` 配置。
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

polyhedral region/uniform RRT*（同一场景，square × 2 即相邻 1 m，默认 switch 调度 0.5 → 0.2，anytime 2500 次迭代，planner seed 7，单次实测，env-rebuilt）：

| 地图 | 首次到达迭代 / T_first / 首解前 N_query | C_first → C_2500 / 最小 clearance | 节点 | 夹障碍路径节点 / 路径节点 | 规划耗时 |
| --- | --- | --- | --- | --- | --- |
| `random_circles` | 51 / 0.049 s / 52 | 11.225 → 11.225 / 0.156 | 1963 | — / 12 | 2.96 s |
| `single_post` | 44 / 0.046 s / 45 | 11.666 → 11.582 / 0.067 | 2011 | — / 15 | 2.45 s |
| `post_fence` | 45 / 0.047 s / 46 | 12.431 → 11.364 / 0.150 | 2084 | — / 18 | 2.81 s |

每个 cell 平均约 7 个扩展方向（guard 约 3–5、obstacle facet 约 2–4）。被拒的主要是已被覆盖（9–13 千）和旋转裕度 ≤ 0（1–4 千），`far` 为 0；region 节点 230–370 个 / 430–550 次 region 迭代。random_circles 有 3 次 `rejected_collision`（新 cell 半径 ≤ 0，来自近平行行合并时的 offset 下调），不影响正确性。

v1（多 seed 预建 cell）→ v2：每次迭代至多 1 个 cell，N_query = 构造 cell 数 ≈ 迭代数；二级剪枝后每个 cell 平均只剩 1–2 行。

p_region 消融（`scripts/ablate_region_sampling.py --seeds 20`，每配置 20 个 planner seed，全部成功；完整表格与图见 `results/region_ablation/README.md`）。facet 几何扩展，中位数（括号内为上一版按边界来源分类、`ffe0d5b`）：

| 地图 | 指标 | p=0 | p=0.2 | p=0.4 | p=0.6 | p=0.8 | p=1 | switch | exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `random_circles` | 首解迭代 | 232 | 119 (97) | 62 (66) | 50 (67) | 35 (47) | 30 (37) | 52 (63) | 55 (74) |
| | C_first | 13.33 | 12.40 (12.54) | 12.26 (12.54) | 12.05 (12.69) | 12.21 (12.50) | 12.63 (12.58) | 11.96 (12.62) | 12.13 (12.44) |
| | C_2500 | 12.58 | 11.88 (12.15) | 11.88 (12.01) | 11.91 (12.04) | 11.83 (11.89) | 12.13 (12.02) | 11.68 (12.04) | 11.85 (12.05) |
| `single_post` | 首解迭代 | 103 | 65 (76) | 38 (59) | 30 (50) | 22 (41) | 20 (33) | 37 (52) | 37 (54) |
| | C_first | 11.93 | 11.64 (12.56) | 11.95 (12.91) | 11.82 (12.82) | 11.70 (12.93) | 11.76 (12.95) | 11.78 (12.79) | 11.63 (12.44) |
| | C_2500 | 11.46 | 11.43 (11.64) | 11.14 (11.57) | 11.38 (11.60) | 11.29 (11.70) | 11.14 (11.69) | 11.30 (11.63) | 11.21 (11.72) |
| `post_fence` | 首解迭代 | 119 | 104 (133) | 70 (100) | 53 (179) | 46 (224) | 38 (355) | 57 (130) | 57 (145) |
| | C_first | 11.52 | 11.73 (12.26) | 12.00 (12.27) | 11.66 (12.31) | 11.97 (12.64) | 11.79 (12.84) | 11.71 (12.95) | 11.63 (12.38) |
| | C_2500 | 11.15 | 11.11 (11.61) | 11.25 (11.51) | 11.11 (11.73) | 11.10 (11.83) | 11.15 (12.06) | 11.43 (12.08) | 11.11 (11.84) |

p=0 与上一版完全相同（纯 uniform）。首解迭代数随 p_region 单调下降，三张地图都是 p=1 最快；post_fence 不再在 p ≥ 0.6 时变慢（首解迭代 355 → 38）：沿柱子 facet 切向滑动并选择放松约束的转向，region 通道能自己穿过栅栏。C_first 与 C_2500 普遍比上一版好 0.3–1.2。T_first 中位数 0.024–0.19 s（p > 0）；总规划时间受并行负载影响波动较大（p=0 结果相同但耗时差约 20%），不作比较。switch / exp 仍是折中默认值。

orientation 的 clearance 是展开路线上的局部 guarded clearance，一阶 / 二阶是认证路线上稠密采样的 guarded clearance，两者口径不同。对全部树边稠密检查，这些场景中一阶 cell 也没有产生碰撞边；但一阶 cell 内确实存在真实位移超过 `rho` 的构型，只有二阶 cell 是证书。`formation/distance_field.py:61-68` 的纯 Python 行列扫描约 0.6s，是独立热点，当前未改。

## 验证命令与最近记录

最近验证记录：focused 39 tests（`test_polyhedral_frontier` 19 + `test_tube_rrt` 12 + `test_maps` 8），完整 86 tests；这些是最近一次验证记录，不是永久保证。

项目使用的 Conda 环境是 `../env-rebuilt`（即 `/home/eai/projects/env-rebuilt`，可用 `conda activate /home/eai/projects/env-rebuilt` 激活）。

```bash
../env-rebuilt/bin/python -m unittest tests.test_polyhedral_frontier tests.test_tube_rrt tests.test_maps
../env-rebuilt/bin/python -m unittest discover -s tests
for map in random_circles single_post post_fence; do
  for cell in orientation first_order second_order polyhedral; do
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

polyhedral cell + frontier 搜索位于本地分支 `polyhedral-frontier`（基于 `path-plan` 的 `1d6ec45`），只做了本地提交，未推送。下面是更早的 `curve-band-v2` 交付说明，保留备查。

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
