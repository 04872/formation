整个方法位于普通 point/region RRT\* 与 full continuous-ownership RRT\* 之间。Full continuous ownership 的问题在于需要为一个 cell 内不同状态维护不同的最优 predecessor，

\[
V_i(x),\qquad \pi_i(x),\qquad x\in C_i,
\]

进而产生 partial dominance、ownership partition、PWA value 和 region splitting 等复杂结构。这里不采用这一步，但也不把一个已经构造出的 certified cell 再退化成单个 seed。每个树节点定义为

\[
\boxed{
N_i=(q_i,C_i,e_i,J_i,p_i),
}
\]

其中 \(q_i\) 是生成 cell 的 seed，\(C_i\subset C_{\rm free}\) 是对应的 certified convex cell，\(p_i\) 是唯一 parent，\(e_i\in C_i\) 是当前 tree branch 进入该 cell 的连续 entry state，\(J_i\) 是从起点到 \(e_i\) 的 cost。一个 cell 始终只有一个 parent 和一个 entry，因此仍然保持 RRT\* 的 tree semantics；但由于 \(C_i\) 整体已经被认证安全，从 \(e_i\) 出发，cell 内任意状态

\[
x\in C_i
\]

都可以作为当前 branch 的连续可达状态。当前节点因此在整个 cell 内自然诱导出一个简单的单-parent cost cone

\[
\boxed{
\widehat V_i(x)=J_i+d_W(e_i,x),\qquad x\in C_i.
}
\]

这不是 full continuous value field，因为它没有在不同 parent 之间取 lower envelope；它只是表达当前唯一 parent 进入 \(C_i\) 后，在已认证区域内部移动到 \(x\) 的代价。

这一连续化不是只用于 Steer。它同时作用于 **launch、cell-to-cell connection、ChooseParent 和 Rewire**。这是整个方法与“单纯把随机样本投影到 cell”之间最重要的区别。

给定随机样本

\[
q_{\rm rand},
\]

RRT 的全局随机探索机制保持不变。哪一个树节点获得该 sample，可以继续采用当前已有的 nearest 机制；这一部分不是本方法必须修改的内容。选中 \(N_i\) 后，普通 point RRT\* 会固定从 \(q_i\) 开始扩展，而这里允许在

\[
X_i^{\rm launch}
=
C_i\cap B_W(e_i,L_{\rm in})
\]

内连续选择 launch state。\(L_{\rm in}\) 用于限制一次扩展在已有 certified cell 中允许进行的内部移动距离，避免把一个非常大的 cell 当成零成本瞬移区域。

如果完全不利用额外 opening 信息，那么这一连续 launch 的基本解就是随机样本在 launch domain 上的投影：

\[
\boxed{
x_{\rm proj}
=
\arg\min_{x\in X_i^{\rm launch}}
d_W(x,q_{\rm rand}).
}
\]

它的意义不是一个简单 baseline，而是 single-entry continuous-cell 思想的第一层实现：既然整块区域已经安全，就先在这块区域内部走到最接近本次随机样本的位置，然后才真正向未知空间进行新的 Steer。因此 point RRT 中

\[
q_i\rightarrow q_{\rm rand}
\]

被替换为

\[
e_i
\rightsquigarrow
x_{\rm proj}
\rightarrow
q_{\rm rand}.
\]

其中第一段完全处于 \(C_i\) 内，不需要重新探索已经认证的自由空间。

现在加入 constraint-release 信息，但它不再取代 continuous-cell projection，也不再通过临时构造邻域 cell 计算。当前 cell 的每张 obstacle-induced facet 都来自一个明确的 robot–obstacle pair \(k\)。设该 pair 的分离法向为 \(n_k\)，对应 formation slot 为 \(s_{r_k}\)。对 configuration-space 微小运动

\[
d=(v_x,v_y,\omega),
\]

该 robot 的一阶运动为

\[
\dot p_{r_k}
=
v+\omega R(\theta_i)J s_{r_k}.
\]

因此当前 constraint 沿 \(d\) 的 clearance 一阶变化可直接写成

\[
\boxed{
\sigma_k(d)
=
n_k^\top
\left(
v+\omega R(\theta_i)Js_{r_k}
\right).
}
\]

采用 clearance 增大为正的符号约定后，

\[
\sigma_k(d)>0
\]

表示当前 limiting constraint 沿该方向正在释放，而

\[
\sigma_k(d)<0
\]

表示它正在进一步收紧。这个量直接来自当前 cell 已经拥有的 proximity information，不需要在 \(q_i\pm h e_j\) 重新构造任何临时 cell。

单纯知道一个 constraint 正在释放仍然不够，因为它失效后可能马上有另一个 constraint 接管。于是对每个当前 obstacle facet \(k\) 定义 unlock leverage。记其 configuration-space outward direction 为 \(u_k\)，当前 cell 沿该方向的 support 为

\[
\ell_i(u_k)
=
\max_{\xi\in C_i}u_k^\top\xi.
\]

然后只从**当前 cell 已经存在的约束系统中**暂时去掉第 \(k\) 个 obstacle constraint，保留其他所有 obstacle constraints 和 guard，计算

\[
\ell_i^{-k}(u_k).
\]

定义

\[
\boxed{
r_k=
\left[
\ell_i^{-k}(u_k)-\ell_i(u_k)
\right]_+.
}
\]

这里没有构造另一个 seed 对应的新 cell；它只是问：在当前这组约束中，如果 facet \(k\) 不再限制 region，后面的其他约束距离它还有多远。若 \(r_k\approx0\)，则该 facet 即使释放也几乎不会打开额外空间；若 \(r_k\) 较大，则说明它确实挡住了一块潜在自由空间。由于 \(r_k\) 只依赖当前 cell，可以在 cell 创建时计算并缓存，不需要每次随机扩展重新求。

本次随机样本给出探索方向 \(d_s\)。于是定义每个当前 facet 的 release utility

\[
\boxed{
U_k(d_s)
=
r_k[\sigma_k(d_s)]_+.
}
\]

它同时表达“这个约束正在多快地释放”和“释放以后实际有多少空间可以打开”。整个 cell 的 release signal 可以取

\[
G_i(d_s)
=
\sum_kU_k(d_s),
\]

而最值得关注的限制 facet 为

\[
k^\star=\arg\max_kU_k(d_s).
\]

这里的 constraint-release 信息不是用来重新决定 RRT 的全局探索方向，而是作为 continuous-cell launch 的**辅助信息**。这一点根据目前 projection 比 release-only 更稳定的实验结果尤其重要：RRT sample 本身已经携带有效的随机探索偏置，因此 release information 不应该优先级高到把 launch point 任意拉离 sample-directed projection。

因此综合后的 launch rule 采用“projection first, release as a constrained correction”。先求 \(x_{\rm proj}\)，再定义一个 near-projection continuous set

\[
\boxed{
X_i^\varepsilon
=
\left\{
x\in X_i^{\rm launch}:
d_W(x,q_{\rm rand})
\le
d_W(x_{\rm proj},q_{\rm rand})+\varepsilon_{\rm prog}
\right\}.
}
\]

这个集合表示：只允许 opening information 在“对随机样本推进几乎同样好”的状态之间进行选择，而不能为了追逐某个 release facet 明显牺牲 RRT 原有的 sample-directed exploration。

如果所有 \(U_k\) 都很弱，则直接使用

\[
x_i^\star=x_{\rm proj}.
\]

如果存在明显的 release signal，则在 \(X_i^\varepsilon\) 内利用最重要 facet \(k^\star\) 选择 launch point。可以先最大化该 facet 的 progress：

\[
\alpha^\star
=
\max_{x\in X_i^\varepsilon}
a_{k^\star}^\top x,
\]

然后在

\[
a_{k^\star}^\top x
\ge
\alpha^\star-\varepsilon_f
\]

的近最优集合中再最大化 sample-direction progress

\[
d_s^\top x.
\]

于是得到

\[
\boxed{x_i^\star.}
\]

因此这里不存在

\[
\lambda_1\text{ sample distance}
+
\lambda_2\text{ opening score}
\]

这种人为加权。决策层次是明确的：

\[
\boxed{
\text{首先保持接近 projection 的 RRT 探索效果，
然后才利用 constraint release 在这些近等价 launch states 中进行修正。}
}
\]

这部分是当前 constraint-release 指标加入原 single-entry continuous-cell 方案后唯一新增的 launch 机制。它不是 continuous-cell 方法本身的全部内容。

确定 \(x_i^\star\) 后，从

\[
x_i^\star
\]

向 \(q_{\rm rand}\) 执行现有 Steer，得到新的 seed

\[
q_{\rm new}.
\]

随后构造

\[
C_{\rm new}=C(q_{\rm new}).
\]

到这里 continuous launch 只决定了**新区域在哪里被发现**。但是新 cell 在树中的 parent、entry 和 cost 不能简单固定为 \(x_i^\star\) 或 \(q_{\rm new}\)，因为 single-entry continuous-cell 方法还要求 **cell-to-cell connection 本身也是连续优化的**。

对一个候选 parent cell \(C_j\)，若它与 \(C_{\rm new}\) 的合法连接集合为

\[
P_{j,\rm new}
=
C_j\cap C_{\rm new},
\]

则 parent \(j\) 通过当前 entry \(e_j\) 对该 portal 中任意状态 \(z\) 的到达代价为

\[
J_j+d_W(e_j,z).
\]

因此，parent \(j\) 对新 cell 提供的最优 continuous attachment point 为

\[
\boxed{
e_{\rm new}^{(j)}
=
\arg\min_{z\in P_{j,\rm new}}
\left[
J_j+d_W(e_j,z)
\right].
}
\]

相应 candidate cost 为

\[
\boxed{
J_{\rm new}^{(j)}
=
J_j+d_W(e_j,e_{\rm new}^{(j)}).
}
\]

这一步就是原 single-entry continuous-cell 方案中的 **continuous portal attachment**。它与 launch point 是两个不同概念：

\[
x_i^\star
\]

决定“从哪里向未知空间扩展并发现 \(C_{\rm new}\)”；

\[
e_{\rm new}^{(j)}
\]

决定“如果最终选 parent \(j\)，从 parent region 的哪个连续状态进入新 region 成本最低”。

二者不要求相同。

随后 RRT\* 的 ChooseParent 也在 continuous region level 上执行。对所有通过现有 NearConnect / overlap test 的候选 parent \(j\)，分别求

\[
e_{\rm new}^{(j)},\qquad
J_{\rm new}^{(j)},
\]

然后选择

\[
\boxed{
j^\star
=
\arg\min_jJ_{\rm new}^{(j)}.
}
\]

最终设置

\[
p_{\rm new}=j^\star,
\]

\[
e_{\rm new}=e_{\rm new}^{(j^\star)},
\]

\[
J_{\rm new}=J_{\rm new}^{(j^\star)}.
\]

因此这里的 ChooseParent 不再比较 seed-to-seed edge cost，而比较**不同 parent region 通过各自最优 portal point 到达新 region 的连续 cost**。

Rewire 同样需要保持这一连续语义。对于新节点 \(N_n\) 和附近已有节点 \(N_i\)，若

\[
P_{ni}=C_n\cap C_i
\]

是合法连接 portal，则计算

\[
\boxed{
z_{n\to i}^\star
=
\arg\min_{z\in P_{ni}}
\left[
J_n+d_W(e_n,z)
\right].
}
\]

如果

\[
J_n+d_W(e_n,z_{n\to i}^\star)
<
J_i,
\]

则不是只改变一条离散 edge，而是执行 whole-cell rewire：

\[
\boxed{
p_i\leftarrow n,\qquad
e_i\leftarrow z_{n\to i}^\star,\qquad
J_i\leftarrow J_n+d_W(e_n,z_{n\to i}^\star).
}
\]

这里依然没有 partial ownership：整个 \(C_i\) 统一更换 parent 和 entry，因此仍然是一棵普通意义上的树。

但这一点带来一个不能省略的后果：**如果 \(e_i\) 因 rewire 改变，那么 \(i\) 的 children 原来保存的最优 portal entries 一般也会失效。** 对 child \(c\)，原来

\[
e_c
=
\arg\min_{z\in C_i\cap C_c}
[J_i+d(e_i,z)]
\]

是针对旧的 \(J_i,e_i\) 求得的。rewire 后必须重新求

\[
e_c',
\qquad
J_c',
\]

然后继续影响其 descendants。

从方法语义上，正确操作是递归执行 descendant repair：

\[
N_i\text{ rewired}
\Rightarrow
\text{recompute entries/costs of its descendants}.
\]

为了避免一次 rewire 导致整个 subtree 立即重新优化，工程实现可以使用 lazy versioning：节点 \(i\) 的 \((e_i,J_i)\) 更新后增加 version，并把 descendants 标记为 stale；只有当某个 descendant 下一次需要参与 ChooseParent、Rewire、扩展或者最终 path extraction 时，才沿 parent chain 刷新其 portal attachment 和 cost。这里 lazy repair 只是执行方式，数学上的 tree state 仍定义为最新连续 attachment 的结果。

最终从 goal node 回溯时，树路径不是 seed sequence

\[
q_0,q_1,\ldots,q_m,
\]

而是连续 entry sequence

\[
e_0,e_1,\ldots,e_m.
\]

因为

\[
e_{i+1}\in C_i\cap C_{i+1},
\]

所以

\[
e_i\rightarrow e_{i+1}
\]

可以在 \(C_i\) 内安全连接。于是 tree 本身已经给出一条由连续 portal states 构成的可行路径。随后仍然可以在整条 tube

\[
C_0\rightarrow C_1\rightarrow\cdots\rightarrow C_m
\]

内部再做最终的 joint path optimization；那一步是全路径联合优化，而这里的 entry/ChooseParent/rewire 都只是 RRT\* 搜索过程中逐节点、增量地维护单-parent continuous state，因此两者不是同一个过程。

所以完整的方法关系不是

\[
\text{projection}
\equiv
\text{continuous-cell RRT*},
\]

而是

\[
\boxed{
\begin{aligned}
\text{Single-entry continuous-cell RRT*}
={}&
\text{continuous launch domain}\\
&+\text{continuous portal attachment}\\
&+\text{continuous ChooseParent}\\
&+\text{continuous whole-node Rewire}\\
&+\text{descendant continuous-entry repair}.
\end{aligned}
}
\]

其中 simple projection

\[
x_{\rm proj}
\]

只是 **continuous launch domain 的基础选择方式**。

而当前 constraint-release 机制

\[
U_k(d_s)=r_k[\sigma_k(d_s)]_+
\]

是建立在这套完整 continuous-cell RRT\* 之上的**可选 exploration enhancement**。它只回答：

\[
\text{在已经允许连续选择 launch state 的前提下，
当前 cell 的 constraint structure 是否提供了比纯 projection 更多的 opening 信息？}
\]

它不替代 continuous portal attachment、ChooseParent 或 Rewire。

因此当前最完整的算法流程应当是

\[
\boxed{
\begin{aligned}
q_{\rm rand}
&\rightarrow \text{现有 Nearest / node selection}\\
&\rightarrow C_i\text{ 内 continuous launch domain}\\
&\rightarrow x_{\rm proj}\\
&\rightarrow \text{可选 constraint-release correction}\\
&\rightarrow x_i^\star\\
&\rightarrow \operatorname{Steer}(x_i^\star,q_{\rm rand})\\
&\rightarrow C_{\rm new}\\
&\rightarrow \text{NearConnect candidates}\\
&\rightarrow \text{每个 parent 的 continuous portal attachment}\\
&\rightarrow \text{continuous ChooseParent}\\
&\rightarrow \text{whole-cell continuous Rewire}\\
&\rightarrow \text{lazy descendant entry/cost repair}.
\end{aligned}
}
\]

其中前面目前已经实测的 `projection` 和 `release-guided launch`，只覆盖了这条链的 **launch 部分**。它们的 \(9/9\) 和 \(8/9\) 结果还不能代表完整 single-entry continuous-cell RRT\* 的效果，因为 continuous ChooseParent、continuous portal attachment 和 continuous Rewire 尚未包含在那个比较里。

基于目前实验，主方案应该把 **single-entry continuous-cell RRT\*** 作为核心，把 constraint-release 作为待验证的增强，而不是反过来。也就是说，即使未来发现 release correction 没有收益，完整方法仍然保留

\[
(C_i,e_i,J_i,p_i)
\]

以及连续 launch、continuous attachment、continuous parent selection、continuous rewire；只需令

\[
x_i^\star=x_{\rm proj}
\]

即可。只有实验进一步证明 release information 在某些 projection 容易失败的场景中带来额外收益，才把它作为完整方法中的 opening-aware extension 保留。