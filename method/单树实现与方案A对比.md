# 单树实现与方案 A 对比

本次新增独立的固定锚点单树实现，保留现有方案 A。实验比较整个新算法，不把差异单独归因于内部细化、Nearest 或任一局部机制，也不宣称动态 cover 的渐近最优性。

## 实现与对照身份

- 新实现：`formation/single_tree_rrtstar.py` 的 `SingleTreeRRTStar` / `SingleTreeConfig`。
- 对照入口：`scripts/compare_single_tree.py`，结果目录 `results/single_tree_reference/`。
- 方案 A：当前 `formation/region_coupling.py` 的 `SingleEntryRRTStar`，variant `single_entry_rrtstar`，原始 `PolyhedralCellModel`，步长 **0.3**。不是旧 `RegionTubeRRTPlanner` 的 continuous 模式，也不是 Cost-Near/normal-turn 变体。
- 基线识别依据：`results/cost_near_all/experiment.json`、`scripts/cost_near_summary.py` 的 `BASE=single_entry_rrtstar@0.3`，以及 Cursor 记录中最近一次 Cost-Guided 实验的明确设置。该历史实验为 50 seeds、6000 迭代，每 50 次作 checkpoint，再按在线时间对齐；在线时间排除链优化。本次改为下面明确列出的真实 wall-clock 协议，因此不能把历史表格耗时直接混入本表。
- 运行前仓库 HEAD：`81c3ea4075b818b09a1f2a973bc402c25b40282a`；未改动 A 的实现或默认行为。原有四份未跟踪理论文档保持原样。

新树的每个节点保存固定 pose、固定 owner、parent、children、固定父边 portal 及边长。全部覆盖证书查询后，覆盖样本在 owner 中均匀随机选取并插入；只按完全相同 pose + owner 去重，重复节点会重考虑，不使用效用拒绝或 h 合并。未覆盖样本从同一状态树找 Nearest，复用该节点 owner 作投影与 Steer；成功生成证书后，以严格交叠 witness 为新锚点。源 owner 的可达节点作为回退父，即使超出近邻半径也保留。

新版本的未覆盖样本 Nearest 使用**固定锚点距离**，A 使用原来的**区域 gap 选择**；后续随机数流还会因公平 owner 选择而分叉。所以本次不是仅打开覆盖插点的单变量消融，也不要求两版本生成相同证书集合。若观察到收益，不能全部归因于内部插点；若观察到损失，也不能单独归因于多状态表示。

ChooseParent 和 Rewire 的邻居是 **固定 r0=1.5 m（d_W）内的全部状态**，没有 k-log 或 top-K 截断；配置继承的 `max_neighbors=16` 仅适用于 A。新树的跨 owner 边先查端点是否可作严格 portal，随后检查证书对 witness，并按固定端点求 via；求解不可用时仅保留经复核的可行 witness 上界，绝不使用零代价。边长在 owner 兼容的角度展开 chart 中计算。改父后立即更新整棵后代子树的 g，保持端点/portal 固定；ChooseParent 排除自身及后代，Rewire 排除祖先。没有固定状态图、SSSP 或全局 Di。

新实现默认 `goal_margin=0`，符合草案的普通 Ci 目标域。**本次主实验显式使用 goal_margin=δ=0.02**，与 A 的目标域和共享链优化器一致；因此本表不使用放宽目标域作为收益。两者均为机器人半径 0.113 m、地图 safety margin 0.06 m、额外 safety_distance 0.02 m、portal δ=0.02 m、active_range=1 m、max_extent=1.5 m、goal bias=0.1、初始步长 0.3 m，Steer 最多 4 次回退。安全证书编码的机器人表面间隙至少为地图裕度加额外距离，即 0.08 m；报告的 guarded clearance 另减去地图裕度，应至少为 0.02 m。几何 slack 与精确障碍距离分开处理。

该实现使用动态证书发现。理论文档给出的固定有限 cover、公平持续采样和固定半径全候选下的鲁棒类 limsup 上界，不能直接用于本实验的动态 cover；等号还需额外代价稠密性。未实现/宣称稀疏高效 AO 版。

## 实验协议与指标

两个场景 `forest`、`slots`，固定地图，planner seeds **0–9**。每个场景、seed 中串行运行两个版本，奇偶 seed 交换执行顺序，不让规划器互相争抢计算资源。每个 run 持续约 8 s，从同一历史取 2 s、8 s 两个预算；没有根据结果调参。

计时从 planner 构造前开始，包含初始化、迭代、路径提取、链优化及稠密 clearance 验证。首次出现目标以及随后约每 0.25 s 作 checkpoint；只有**完整计算并验证结束时间不超过预算**的输出计入该预算。临界时刻完成的 checkpoint 可能略超预算，原始数据保留但不用于该预算的 best。最后一次预算内观测通常约为 1.75/7.75 s，表中给实际完成时刻；没有把预算外的最终求解算作预算内结果。`actual_wall_s` 记录实际结束耗时（可有一个不可中断操作的超时）。JSON 序列化及生成统计图不计入规划耗时。

- **raw**：当时单树父链的真实可行代价；A 为其 mutable-entry 父链代价，新树为固定 anchor/portal 父链代价。
- **+chain**：两者对各自 owner 链使用同一个 `RegionGeometry.chain` 优化器、相同 portal 和 goal margin；新树只压缩相邻重复 owner，非相邻重复 owner 保留，链接口逐段展开 yaw，允许再次进入同一证书。
- **first**：首个有效输出；`first_seen_s` 另记首次发现可达目标的时刻。首解输出耗时包含后处理和验证。
- **best**：预算内已输出的历史最小代价，保存对应路径；**current** 为最后有效 checkpoint 的路径，不把某个更早的 best 冒充当前树输出。
- 首解后改善按每个成功 run 的 `(first-best)/first` 计算后取中位数。raw 和 +chain 分开计算。
- 运行异常、无解和验证失败不会从成功率分母删除。配对成本只用相同预算下**共同成功**的 seeds。内部 solver 错误与 checkpoint 验证失败单独保留；可恢复的 solver 错误不冒充零代价。
- 每次有效 checkpoint 验证 +chain 路径，新树还验证原始树路径；guarded clearance 低于 0.02 m（数值容差 1e-6）则该 checkpoint 失败。采样步长为 d_W 0.01 m，这是数值稠密检查，另有 Ci/portal 证书，而非新的连续碰撞形式证明。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../env-rebuilt/bin/python -m pytest \
  tests/test_single_tree_rrtstar.py tests/test_region_coupling.py -q
```

结果：**12 passed**。覆盖：低 slack 覆盖样本仍保留、全部覆盖 owner 查询与公平选择、±π chart 接缝代价不低估、跨 owner 端点快检、锚点固定、真实 g/children 一致、防环、树目标 raw 代价不增、portal 严格裕度、clearance 以及半径外同 owner 回退；现有 coupling 测试保持通过。初次普通 pytest 自动加载了环境的 ROS 插件并因缺少 `lark` 失败；通过关闭自动加载隔离了与本任务无关的插件，未安装或修改环境。

## 可复现命令与产物

```bash
../env-rebuilt/bin/python scripts/compare_single_tree.py \
  --scenarios forest slots --seeds 10 --budgets 2 8 \
  --out-dir results/single_tree_reference

../env-rebuilt/bin/python scripts/compare_single_tree.py \
  --scenarios forest slots --seeds 10 --budgets 2 8 \
  --out-dir results/single_tree_reference --analyze-only
```

`experiment.json` 保存命令、解释器、平台和协议，每个 `runs.jsonl` run 另存完整 planner 配置；`environment.json` 保存 HEAD 与依赖版本，`source/` 保存新增代码及测试快照。`runs.jsonl` 包含逐 checkpoint 的时刻、raw/+chain 代价、路径、clearance、节点/证书数及统计，异常保留 traceback；`summary.csv/json` 为分组统计，`paired.csv` 为共同成功配对汇总，`paired_runs.csv` 为逐 seed 配对成本，`paired_cost_changes.png` 显示种子间变化。`results/single_tree_pilot/` 为正式实验前单 seed 的 1/2 s 可运行检查，不混入正式统计。结果目录受仓库 `.gitignore` 排除，但文件在本地主机保留。

## 实测结果

所有表格中，代价单位为 d_W 的 m；组内数值为中位数，最小 clearance 除外。配对变化先按每个共同成功 seed 计算，再取中位数；正值表示新树更长。**正式 40 个 run 均无运行异常、solver 错误或路径验证失败。** slots 的短预算失败是尚未找到解，仍计入成功率分母。

|场景|预算/s|A 成功|新树成功|共同成功|raw best 变化|+chain best 变化|raw 更好/更差|+chain 更好/更差|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|forest|2|10/10|10/10|10|+0.42%|+1.17%|4/6|3/6|
|forest|8|10/10|10/10|10|+11.03%|+9.98%|1/9|2/7|
|slots|2|10/10|7/10|7|-1.13%|-1.08%|4/2|4/3|
|slots|8|10/10|10/10|10|+2.66%|+2.76%|2/8|2/8|

更好/更差以 ±0.1% 为阈值，其余视为近似相同。slots 在 2 s 的共同成功子集偏向较容易的 runs，不能将子集约 1% 的路径缩短解释为整体改善。

### raw：在线树父链

|场景|预算/s|版本|first|best|current|首解后改善|
|---|---:|---|---:|---:|---:|---:|
|forest|2|A|16.747|16.170|16.170|1.13%|
|forest|2|single_tree|18.495|17.202|17.202|1.70%|
|forest|8|A|16.747|15.183|15.210|8.13%|
|forest|8|single_tree|18.495|16.672|16.672|2.53%|
|slots|2|A|15.428|14.520|14.520|3.80%|
|slots|2|single_tree|16.720|14.918|14.918|11.23%|
|slots|8|A|15.428|14.190|14.236|7.75%|
|slots|8|single_tree|16.560|14.602|14.602|10.75%|

### 同一优化器后的 +chain

|场景|预算/s|版本|first|best|current|首解后改善|
|---|---:|---|---:|---:|---:|---:|
|forest|2|A|16.034|15.528|15.528|0.56%|
|forest|2|single_tree|17.295|16.340|16.344|1.53%|
|forest|8|A|16.034|14.513|14.513|5.97%|
|forest|8|single_tree|17.295|15.881|15.881|2.64%|
|slots|2|A|14.576|13.970|13.983|2.85%|
|slots|2|single_tree|14.463|14.221|14.221|2.85%|
|slots|8|A|14.576|13.591|13.634|5.56%|
|slots|8|single_tree|14.524|13.938|13.964|3.45%|

### 首解、规模与实际时间

|场景|预算/s|版本|首次目标发现/s|首个有效输出/s|最后预算内输出/s|节点|证书|
|---|---:|---|---:|---:|---:|---:|---:|
|forest|2|A|0.075|0.086|1.759|1636|1636|
|forest|2|single_tree|0.167|0.181|1.765|1164|952|
|forest|8|A|0.075|0.086|7.760|4734|4734|
|forest|8|single_tree|0.167|0.181|7.766|2976|2150|
|slots|2|A|0.135|0.141|1.756|1192|1192|
|slots|2|single_tree|0.118|0.126|1.759|1598|691|
|slots|8|A|0.135|0.141|7.756|2590|2590|
|slots|8|single_tree|0.281|0.290|7.761|4192|1268|

首解时间按该预算内成功的 runs 统计，故 slots 新树的 2 s 行只含 7 次成功，而 8 s 行包含 10 次；两个中位数不可当作同一子集。2 s 的结果来自 8 s run 的历史，完整实际结束耗时如下。

|场景|版本|完整实际 wall/s|best 输出 guarded clearance 最小值/m|
|---|---|---:|---:|
|forest|A|8.010|0.040002|
|forest|single_tree|8.015|0.040002|
|slots|A|8.006|0.040000|
|slots|single_tree|8.010|0.040007|

上表 clearance 为各 run 的预算内 best +chain 路径的最小值；全部 checkpoint 以及新树原始路径也通过了相同阈值检查。新树 raw 目标代价在所有已记录相邻有效 checkpoint 间均未上升；+chain 的 current 可以比历史 best 长，表中分别保留。

### 8 s 预算内的耗时分解

|场景|版本|Nearest/Projection|全部覆盖查询|Steer|连接/重连|全部checkpoint|其中chain优化|
|---|---|---:|---:|---:|---:|---:|---:|
|forest|A|1.650|—|2.802|2.864|0.318|0.158|
|forest|single_tree|2.493|0.218|1.151|3.420|0.438|0.156|
|slots|A|2.497|—|3.480|1.354|0.185|0.076|
|slots|single_tree|2.266|0.197|0.790|4.156|0.274|0.065|

单位均为 s，取各 run 最后预算内 checkpoint 的累计量中位数。Nearest/Projection 同时包括选源和源证书投影；Steer 不包含投影。连接/重连含子树传播，chain 时间已包含在 checkpoint 总时间内，二者不能再相加。未列出的采样、存储、距离数组等开销仍包含在 wall-clock 中。

在 forest，新树的选源/投影为 2.493 s、连接/重连为 3.420 s，是可见的主要开销；子树 g 传播仅 0.0062 s。slots 的传播为 0.0153 s。不能把连接成本笼统解释为“代价传播/树修复太贵”，当前测量并不支持传播是瓶颈。

## 结论与局限

**本次固定半径全候选参考实现可以运行并保持所检查的单树不变量，但在当前 10 seeds、2/8 s 的实验中，没有表现出对 A@0.3 的整体优势。**

- forest 的 8 s 结果中，新树 raw 历史 best 配对中位数长 11.03%，共享链优化后仍长 9.98%；raw 只有 1/10 对更好，+chain 为 2/10。首解也更慢，8 s 内持续改善弱于 A。
- slots 的 2 s 新树成功率为 7/10，A 为 10/10；8 s 时两者都为 10/10，新树 raw/+chain 配对中位数仍长 2.66%/2.76%。新树 raw 首解后改善幅度较大，但它从更差的初始父链出发，最终没有超过 A。
- 新树确实保留了大量共享 owner 的状态；slots 在相同预算下的状态数更多、证书数更少。耗时分解同时显示其连接/重连负担明显增加。这是实现了持续细化的证据，不是有效收敛速度或优越性的证明。
- 新树 Nearest 也改成了固定 anchor 距离，并使用固定半径全候选；A 仍为区域 gap 选择和至多 16 个证书邻居。故本次是整体方案对比，不能从结果识别单个机制的因果作用，也未证明所有单树变体都不如 A。
- 只有两个场景、10 个匹配 seeds 和短/长两个有限预算；没有以失败结果调参，也不据此推断无限时间表现。固定有限 cover 的理论结论、动态证书生成覆盖性、实用收敛速度仍是不同问题。

![逐 seed 配对成本变化](../results/single_tree_reference/paired_cost_changes.png)

