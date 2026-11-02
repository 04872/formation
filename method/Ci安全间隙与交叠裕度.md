# $C_i$ 安全间隙与交叠裕度

> 本说明按本次最新 v0.2 规则，不继承旧稿全局 $D_i$ 假设。$C_i$ 是代码构造的安全证书；portal $\delta$ 是局部连接条件。

## $C_i$ 的安全来源

`formation/region_coupling.py` 中 `_CorePlanner.__init__`（约 495 行）设置

$$margin=robot\_radius+map.safety\_margin.$$

`PolyhedralCellModel.proximity`（约 270 行）进一步减去 `clearance_margin`；`make_cell`（约 284–289 行）设置 `offsets=clearance-safety_distance`，inactive guard 同样减 `ds`；lazy rows（约 321–325 行）沿用该规则。因此 $C_i$ 证书编码了机器人半径、地图安全裕度及额外 `safety_distance`（默认 0.02）。在代码采用的圆机器人/障碍表示和相应数值认证假设下，实际机器人几何与障碍的距离保证至少为 map safety margin 加额外 $ds$。这项解释受模型及求解容差约束，不将 polyhedral slack 当作欧氏距离。

`RegionGeometry.slack`（约 213 行）汇总 active rows、guard、yaw chart 等约束；它不是精确欧氏障碍距离，也不是统一归一化的内切球半径。边界计算使用的数值容差须独立记录，不构成物理安全裕度。

## 三种 margin 不可混为一谈

| 名称 | 作用 | 是否 $C_i$ 安全所需 |
|---|---|---|
| map `safety_margin` 与额外 `safety_distance` (`ds`) | 随机器人半径收缩 cell 约束，形成 $C_i$ 安全证书 | 是，按当前构造 |
| portal `delta`（默认 0.02） | `via` 仅对公共 portal $z$ 增加两侧 slack 要求 | 否；它是严格 owner 转换条件 |
| 数值容差 | 几何边界、优化器可行性判断 | 属于数值处理，不是物理距离 |

`RegionGeometry.max_margin`（约 372 行）求最大公共 slack；`via`（约 397 行）只对 portal $z$ 施加 margin，不要求固定端点 $a,b$ slack 达到 $\delta$。`_overlap`（约 640 行）检查公共 witness $w$ 的两侧 slack 至少为 $\delta$。由于各 $C_i$ 凸，若 $x_u,z\in C_u$ 且 $z,x_v\in C_v$，两段线段分别留在对应安全证书内；端点不需另有 $\delta$ slack。

`goal_yaw(..., margin)`（约 228 行）的 margin 会进一步压缩目标 yaw 可行域，不能与 portal margin 或物理 clearance 混同。旧方案若同时使用 portal 与 goal margin $\delta$，新实验需说明是否保留目标域收缩。

## 设计与比较规则

本轮可移除全局 $D_i$ 与 margin 带跳样逻辑，普通状态直接满足 $x\in C_i$；这不要求删掉局部 portal $\delta$。严格 portal 条件目前用于限定鲁棒 owner 转换类，不代表 $C_i$ 只有在内部 slack 达 $\delta$ 才安全。若后续移除 portal $\delta$，需单独分析 owner 转换的鲁棒性、数值稳定性与历史实验可比性。

比较搜索机制时保持相同物理 clearance 配置：一致设置地图安全裕度、`safety_distance` 和 validity checker。否则扩大可行域本身会改变问题，不能把改善归因为搜索。至少分开对照“移除全局 $D_i$/目标额外收缩”与其他搜索改动。代码行号为当前查阅版本附近位置，后续以函数名定位为准。
