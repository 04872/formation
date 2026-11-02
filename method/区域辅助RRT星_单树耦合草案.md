# 区域辅助 RRT*：单树耦合草案

> 理论基准与本稿工程草案的区别：理论依据见[单树细化与路径逼近理论依据](单树细化与路径逼近_理论依据.md)。其条件性结论针对固定有限已认证证书集合、固定半径 $r_0$、全候选近邻和公平持续采样；一般只得 $\limsup c_{best}\le c_{cert}^*$，等号还需额外的代价稠密性假设。本稿当前的 $k_n$、代表稀疏化及动态证书增长仍是实验选项，不因此获得 AO 保证。该文定义的是鲁棒认证路径类代价 $c_{cert}^*$，不是原始自由空间最优代价，也不恢复全局 $D_i$。理论条件是比较基准，不代表已经实现。

> 本文为本轮最新规则 v0.2。$C_i$ 是代码所构造的安全凸证书；本稿不继承旧稿的全局 $D_i$（$C_i$ 内缩域）假设，也不把 portal 的 $\delta$ 条件解释成碰撞安全必需条件。

## 组织与证书

方案待实现、待验证。Nearest、Steer、ChooseParent、Rewire 作用于同一采样树；不维护固定状态图、不运行 SSSP。树节点记为

$$v=(x_v,C_v,parent_v,g_v,parent\_edge_v,children_v),$$

其中锚点 $x_v$ 固定，$x_v\in C_v$，$C_v$ 为可行凸区域证书。多个节点可共享证书。根节点 $x_{start}\in C_{root}$，按普通节点处理。

实现中 $C_i$ 由机器人半径、地图安全裕度及额外 `safety_distance` 收缩得到。因而 $C_i$ 的证书安全性不是“只离障碍 $\delta$”：需结合代码的圆机器人/障碍建模及数值认证假设理解。几何 slack 非负表示满足证书不等式；它不是精确欧氏障碍距离，也不等价于统一归一化内切球半径。数值容差单独处理，不将边界浮点误差说成物理裕度。所有凸性、直线连接和边长计算在证书兼容的角度展开 chart 中进行，不能任意以 wrap 最短角差替代该证书内路径。

## 每轮采样与扩展

每轮从固定正概率的全配置 uniform 分布或 goal bias 取样。Nearest 选树节点，可按 cell gap 取 top-$K$ 后使用

$$g(v)+d(x_v,\Pi_{C_v}(q))+d(\Pi_{C_v}(q),q),$$

也可暂留现有部分几何规则。此量是候选排序值，不自动成为可行总代价或剪枝下界。

对 $q$ 先查询空间索引中的**全部已知证书覆盖**，不能只查 Nearest 的 owner 后把样本跳过。若 $q\in C_i$，选一个能接入当前树的覆盖证书，固定候选 $x_{new}=q$ 并复用 $C_i$。若没有已知证书覆盖，沿原 Projection/Steer 生成新证书，并要求与父证书有严格 overlap witness $w$；令固定锚点 $x_{new}=w$。不另造被跳过样本的 cell，也不要求样本落在 margin 带之外才成为合法节点。

首版不因即时收益拒绝候选，不让 $U$ 筛掉所有内部状态。近邻代表机制只作非默认实验项：验证机制时关闭 $h$ 近似拒绝，仅精确 pose 且同 owner 去重并 `RECONSIDER`。$h_n$ 稀疏规则尚未证明 AO。首版 uniform 与 goal bias 共用同一单树循环，不增加独立细化流。

## 局部 portal 与连接

对相邻证书定义

$$P_{ij}(\delta)=\{z\in C_i\cap C_j: slack_i(z)\ge\delta,\ slack_j(z)\ge\delta\}.$$

跨 owner 连接经 $P_{ij}(\delta)$ 选 portal $z$，路径长为 $d(x_u,z)+d(z,x_v)$。由于 $x_u,z\in C_u$ 且 $z,x_v\in C_v$，凸性保证两段分别留在各自安全证书内；端点 $x_u,x_v$ 无需满足 $\delta$ slack。portal 保存在边而不属于可变节点入口。

首版规则：同 owner 节点才直接用证书内线段；不同 owner 通过 $C_u,C_v$ 的 $P_{uv}(\delta)$。不以“第三个公共证书 $C_k$ 同时包含两端”直接绕过 owner 切换：两端共属 $C_k$ 能证明该条几何线段在 $C_k$ 内安全，却不自动证明从各自 owner 转入 $C_k$ 的路径具有严格 portal。显式构造并验证 owner 切换门户后，才可把公共证书作为后续优化。owner 不同本身不意味着几何不连通。

同 owner 连接无需 solver。先用 metric lower bound 与已知可行 pair witness 上界筛选；必要时再调用 `RegionGeometry.via` 求 portal。该函数的 `margin` 只压缩门户 $z$ 的可行域，不要求固定端点 slack 达到该值。求解失败不能记为零成本。

固定位置查询邻居并保证当前 parent 留在候选集中。令 $n$ 为节点数，可试

$$k_n=\min(n,\lceil k_0\log(n+1)\rceil),$$

单独采用此式不构成 AO 保证。

## 选父、重连与目标

新增节点直接比较 $g_{new}=\min_u\{g(u)+L(u,new)\}$。ChooseParent 排除节点自身及其子孙、保留旧 parent 回退；Rewire 排除自身祖先。改父不移动锚点、不重选入口、不搬动后代边 portal。代价变化沿 children 更新，或按需刷新 parent 链及 goal 候选整棵子树。

目标用 `goal_yaw(..., margin=0)` 选任意落在证书中的合法目标，再由同证书凸性线段认证。不得默认用 `margin=delta` 收缩目标域；若保留旧 goal margin $\delta$，须说明这是目标几何域的额外收缩。旧方案 A 的 portal 与 goal 都用过 margin $\delta$，历史 clearance 观测约 $0.04$ 并不自动证明新变体具有相同保证。

公平实验须按相同物理 clearance 配置 `safety_distance`、地图安全裕度和 validity checker；不能把放宽可行域带来的改善归于搜索机制。应隔离对照“仅移除全局 $D_i$/目标额外收缩”与搜索机制变化。保留严格 portal $\delta$ 是当前兼容连接约束的选择，不等于 $C_i$ 碰撞安全要求；去掉它需另行分析鲁棒连接、数值稳定性与历史实验可比性。

```text
初始化唯一树 T={root}
每个 q:
    查全部已知 C_i 覆盖；若覆盖，取可接入证书并令候选锚点为 q
    否则沿原 Projection/Steer 生成 C_new 与严格 overlap witness w，锚点为 w
    精确重复且同 owner 时 RECONSIDER；否则插节点并 ChooseParent
    Rewire 邻居；刷新受影响后代及 goal 候选
```

## 证明与验证边界

本方案没有已建立的完备性或 AO 保证。固定 $\delta$ 只定义鲁棒 portal 类；区域生成覆盖性、采样/连接条件、代表拒绝、代价传播仍需分析。实现应作为新 variant，不直接继承 `region_coupling.py` 的 mutable-entry `_connect/_refresh` 语义。先做同 walltime 对照，报告 forest、slots 和 clearance 设定，不据此声称全局优势。
