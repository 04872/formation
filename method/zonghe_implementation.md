# Single-entry continuous-cell RRT* 实现

对应最新 [zonghe.md](zonghe.md)。原方案文件未修改。
主体现在是完整 continuous-cell 树维护，constraint release 是受 projection 约束的可选增强。
默认 `RegionTubeConfig.expansion="projection"`；`--expansion opening` 启用增强，
`point` 只作为 seed-launch 消融，`tube` 显式保留旧 TubeSteer。没有删除旧分支。

## 树状态与代价

`TubeRRTNode` 保存 seed、certified cell、唯一 parent、实际 entry 和到 entry 的 cost。
连续模式统一使用 `d_W=sqrt(dx²+dy²+rho²*dtheta²)`，yaw 取当前 cell 的局部 chart。
`J_i` 是到 entry 的长度，节点在 cell 内诱导 `J_i+d_W(e_i,x)`。
旧 `tube` 模式仍使用原有 `d_G=||dc||+rho*|dtheta|` 和可选 margin penalty；
连续模式的 J 按原文仅为 W 长度，不加入 margin penalty。

随机样本、goal bias、原有 point nearest 与 NearConnect 候选范围继续使用。
连续 cost 不再采用 seed-to-seed 距离作 portal attachment 的下界；使用 parent 的 J 作廉价下界。
没有 partial ownership、lower envelope、region splitting 或不同位置的多 predecessor。

## Projection first 的 launch

`CellLaunchSearch.select` 先在完整 `X=C_i∩B_W(e_i,L_in)` 上求随机样本的 W 投影。
默认 `L_in=1.0`。release 弱时直接采用投影，不做 correction。
强信号时加入额外 SOC 球：

```
||z-z_rand|| <= ||z_proj-z_rand|| + progress_epsilon
```

默认 `progress_epsilon=0.05`，严格限定 release 最多牺牲多少随机样本推进距离。
在这个 near-projection 连续集合中，先最大化所选 facet 的 `a_k^T z`，
再加入 `a_k^T z >= alpha-facet_epsilon`，最大化原文物理坐标的 `d_s^T x`。
第二阶段做正确的协向量坐标转换，没有用 W 内积替换物理目标。
默认 `facet_epsilon=1e-4`；这是两阶段近最优带宽，不是改写加权目标。
`progress_epsilon=0` 时该集合退化到唯一 projection，不做 release correction。

强信号最多一次 projection 加两次 support；没有候选离散化、体积积分、有限差分、
局部非凸搜索或 hypothetical cell。内部移动预算仅约束 discovery launch，
不错误地套到后续 parent 的整个连续 portal attachment 上。

release 的 `sigma` 与 `r` 沿用当前 cell 的真实 pair geometry：
`U_k=r_k[sigma_k]_+`，按最大单项 U 是否超过默认 0.02 激活，总 G 只作诊断。
`r` 是完整 support 与仅删除单张平面 facet 行后的 support 差，其他 constraints、guard、yaw 不变。
一条绝对 yaw pair 约束展开成两张 signed facets；利用 cell 的 yaw 反射对称性精确共享 leverage。
第一次使用 cell 时缓存，后续仅做向量运算；rewire 改变 entry 不使这个几何缓存失效。

## Continuous attachment 与 ChooseParent

`CellLaunchSearch.attachment(first,second,entry)` 直接求

```
argmin_z d_W(entry,z),  z in first ∩ second
```

保留两张 cell 的所有 obstacle、guard、yaw 约束，在共同 yaw lift 中求解。
当 parent entry 本身位于 child cell 中时，直接得到零内部距离的精确解。
否则使用四变量 norm-epigraph SOCP：`min t, ||z-entry||<=t`。
这是原文的凸距离问题，不是沿两个 seed 连线选一个 witness，也不缩小 intersection。

Steer 生成新 cell 后，先重新优化 discovery parent 的 attachment；
对 NearConnect 中可能更优的 parent，分别求其最优 continuous portal，比较
`J_parent+d_W(e_parent,portal)`。
最终 entry 是所选 parent 的最优 attachment，**不固定为 discovery launch 或 seed**。
原 Steer 的 launch/线段证书只证明如何发现新 cell，不再作为最终树连接的绕路前缀。

## Whole-cell Rewire 与 lazy descendant repair

Rewire 同样投影新 parent 的 entry 到 cell intersection；若新的 J 更小，整体替换 parent、entry、J。
通过排除新节点的祖先防止成环。

每个节点保存 `version` 与 `parent_version`：

- 节点 entry/J 改变后增加 version；不遍历整棵 subtree。
- `_refresh_node` 先沿 parent chain 从上到下检查版本。
- 发现 parent version 变化时，重新求当前两 cell 的最优 portal，更新 entry、J、certificate 和 version。
- 每个节点参与扩展、ChooseParent、Rewire 或目标代价比较前刷新。
- 最终 path extraction 前刷新；返回结果前 `_refresh_all`，确保输出树也没有 stale entry/cost。

后代修复不是沿旧 portal 传播标量 cost，也不是调整旧 launch 前缀。
Goal 是固定终点，刷新时保持目标状态，只重算最后一段长度；不会把终点投影成任意重叠点。

whole-cell entry 改善可能使某个后代的到达代价增加；单 parent cost cone 没有逐点 dominance 保证。
因此移除了旧版“如果现有 goal 变贵就回滚 rewire”的规则。
`cost_history` 记录当前树中最优 goal 的真实代价，可能上升；不是历史最佳路径的 lower envelope。
本批实验确有这种变化，没有将曲线强行压成单调。

## 路径、数值求解与诊断

路径由 `e_0,e_1,...,e_goal` 构成，相邻 entry 在 parent cell 内连接；不强制经过 seed 或 discovery launch。
长 yaw 段在同一 cell chart 中补一个共线中点，避免跨 wrap 时插值走错方向，不改变长度。
最终可选的 tube joint path optimization 未引入搜索阶段。

全部问题使用同一个 Clarabel 凸求解器。Support 三变量，距离 projection/attachment 四变量。
标准 gap/feasibility 容差 `1e-8`，reduced 容差 `1e-7`，静态 KKT 正则化 `1e-10`。
薄交集的距离问题采用 norm epigraph，避免二次目标在这种交集上停滞。
不接受未求解状态，不添加备用优化器；intersection infeasible 是正常的无连接结果。

日志区分 discovery launch 与 final parent attachment，并记录 projection 距离、实际样本距离、
release utility、facet progress、缓存命中、support 查询数和 launch 残差。
Planner 另记录 attachment 查询数/时间、descendant repairs、最终全部 entries 的交集残差与 cost 一致性误差。
精度按实际约束/目标误差报告，不把状态坐标的误差和求解器 gap 混为一谈。

## 运行及验证

```bash
# 默认执行完整 continuous-cell 核心
MPLCONFIGDIR=/tmp/formation-mpl ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py \
  --map single_post --cell polyhedral_tube --slot-scale 2 --iterations 200

# 同一核心上增加受 projection 限制的 release correction
MPLCONFIGDIR=/tmp/formation-mpl ../env-rebuilt/bin/python scripts/visualize_tube_rrt.py \
  --map single_post --cell polyhedral_tube --expansion opening \
  --launch-progress-epsilon 0.05 --launch-facet-epsilon 0.0001 --iterations 200

MPLCONFIGDIR=/tmp/formation-mpl ../env-rebuilt/bin/python scripts/evaluate_cell_launch.py \
  --planner --seeds 3 --iterations 200 --anytime --out-dir results/cell_launch_continuous
```

沿用已有回归，更新 launch 测试以检查 near-projection 约束；仅新增一个聚焦测试，
以有解析投影解的 detour/subtree 检查 rewire 后 deferred entry 变化和真实 descendant reoptimization。
旧 TubeSteer 测试显式指定 `expansion="tube"`，保持它们原有测试对象。
当前 [结果报告](../results/cell_launch_continuous/README.md) 是完整 continuous-cell 版本；
此前 `cell_launch_release` 的 9/9 与 8/9 仅是旧 launch 比较，不代表完整方法。

## 扩大家族实验补充

新增 7 类、14 个配置的[636 次实验报告](../results/continuous_families/README.md)。
实验确认 release 在错列柱群的局部成功率收益，也揭示默认 L_in=1.0 的大区域停滞，
不能再仅凭此前三个简单场景的 9/9 推断总体优势。当前算法和默认参数在该轮实验中保持不变。
