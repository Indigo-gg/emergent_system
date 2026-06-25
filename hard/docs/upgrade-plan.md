# Hard Mode 升级计划

> 基于 debug-fast 实验分析，系统性解决已发现的问题并提升整体质量

---

## 已修复的问题（v1 Hotfix）

### 1. Chemotaxis 力计算 Bug ✅
**文件**: `src/simulation/step.py`
**问题**: `vm_execute_chemotaxis()` 被调用两次，传入相同参数，导致 `fx_env = fy_env`，产生 45° 对角线力
**修复**: 化趋性公式输出标量大小，方向由营养梯度归一化提供
```python
chemo_magnitude = vm_execute_chemotaxis(...)
grad_len = sqrt(gnx² + gny²)
fx_env = chemo_magnitude * gnx / grad_len
fy_env = chemo_magnitude * gny / grad_len
```

### 2. Novelty Score 全部为 inf ✅
**文件**: `src/novelty/filter.py`
**问题**: archive < k 时返回 `inf`，`is_novel()` 对 inf 无限接纳
**修复**:
- `novelty_score()` 返回 `1000.0`（大有限值）替代 `inf`
- `is_novel()` 拒绝 inf 分数，bootstrap 阶段（≥100）视为新颖

### 3. Grid 填充率极低（3/125）✅
**文件**: `src/evolution/features.py`, `src/evolution/map_elites.py`
**问题**: FFT 归一化 `/n` 导致振幅聚集在 0；固定特征范围不适应数据分布
**修复**:
- FFT 归一化改为 `/√n`
- Grid 范围自适应：每 10 次观测更新 min/max（带 10% 边距）

### 4. 截图数量过少（4/60代）✅
**文件**: `src/main.py`, `src/vlm/judge.py`, `config/fast_debug.yaml`
**问题**: 截图只在 `vlm_worthy` 块内生成；`min_novelty_gap=0.3` 过高
**修复**:
- 传递 `novelty_score` 给 `should_call_vlm()`
- 降低 `min_novelty_gap` 至 0.1
- 新增每 10 代定期截图保存（最佳个体）

---

## 升级计划

### Phase 1: 环境可视化（优先级：高）

**目标**: 让 VLM 和用户都能看到粒子行为与环境的关系

#### 1.1 营养场/废弃场热力图
**文件**: `src/rendering/renderer.py`

新增函数:
```python
def render_environment_overlay(
    position_history, nutrient_field, waste_field,
    cfg, output_path
) -> str:
    """
    渲染环境叠加图：
    - 底层：营养场热力图（绿色系）
    - 中层：废弃场热力图（红色系）
    - 顶层：粒子轨迹（时间编码颜色）
    """
```

**实现细节**:
- 从 `EnvironmentLayer` 提取 `nutrient_field.to_numpy()` 和 `waste_field.to_numpy()`
- 使用双色 colormap：营养=`Greens`，废弃=`Reds`
- 叠加时用 `alpha` 混合（营养 0.3，废弃 0.3，粒子 0.6）

#### 1.2 梯度方向箭头图
**文件**: `src/rendering/renderer.py`

新增函数:
```python
def render_gradient_arrows(
    nutrient_grad_x, nutrient_grad_y,
    waste_grad_x, waste_grad_y,
    cfg, output_path, step_interval=50
) -> str:
    """
    渲染环境梯度箭头图：
    - 绿色箭头：营养梯度方向
    - 红色箭头：废弃梯度方向
    - 箭头长度 ∝ 梯度强度
    """
```

#### 1.3 修改 `render_novelty_package()`
**文件**: `src/rendering/renderer.py`

当前只渲染轨迹图，需要扩展为多面板图:
```
┌─────────────────┬─────────────────┐
│  轨迹密度图      │  环境叠加图      │
│  (现有)         │  (新增)         │
├─────────────────┼─────────────────┤
│  营养场分布      │  废弃场分布      │
│  (新增)         │  (新增)         │
└─────────────────┴─────────────────┘
```

#### 1.4 修改主循环传递环境数据
**文件**: `src/main.py`

在 VLM-worthy 块和定期截图块中，需要传递环境场数据:
```python
# 获取当前环境场快照
nutrient_np = environment.nutrient_field.to_numpy()
waste_np = environment.waste_field.to_numpy()
grad_nut_x_np = environment.grad_nut_x.to_numpy()
grad_nut_y_np = environment.grad_nut_y.to_numpy()

package = render_novelty_package(
    genome, features_15d, position_history,
    feature_timeseries, cfg, screenshot_dir,
    nutrient_field=nutrient_np,  # 新增
    waste_field=waste_np,        # 新增
    grad_fields=(grad_nut_x_np, grad_nut_y_np, ...)  # 新增
)
```

---

### Phase 2: 公式多样性提升（优先级：高）

**目标**: 避免进化收敛到常数公式（如 `-2.2497`）

#### 2.1 适应度函数增加公式复杂度奖励
**文件**: `src/evolution/gep.py`

当前适应度:
```python
fitness = (0.4 * f_survival + 0.3 * f_energy_var + 0.3 * f_structure) * speed_factor
```

建议改为:
```python
# 公式复杂度奖励（鼓励探索非平凡公式）
tree = decode_gene(genome.potential_gene, genome.head_length)
formula_depth = tree.depth()
formula_size = tree.size()
f_complexity = min(formula_depth / 8.0, 1.0) * 0.1  # 最多 10% 加成

# 常数公式惩罚（惩罚退化为常数的公式）
if is_constant_formula(tree):
    f_complexity = 0.0

fitness = (0.35 * f_survival + 0.25 * f_energy_var + 0.25 * f_structure + 0.15 * f_complexity) * speed_factor
```

#### 2.2 常数公式检测
**文件**: `src/evolution/gep.py`

新增辅助函数:
```python
def is_constant_formula(tree: Node) -> bool:
    """检测公式是否退化为常数（如 -2.2497, sin(3.14) 等）"""
    if isinstance(tree, Const):
        return True
    # 检查是否所有终端都是常数
    terminals = get_terminals(tree)
    return all(isinstance(t, Const) for t in terminals)
```

#### 2.3 种群初始化增加公式复杂度下限
**文件**: `src/evolution/genome.py`

当前 `random_genome()` 可能生成过短的公式。建议:
```python
def random_genome(cfg, rng, min_depth=3):
    """生成随机基因组，确保最小复杂度"""
    # ... 现有逻辑 ...
    # 检查深度，如果太浅则重新生成
    while tree.depth() < min_depth:
        gene = _random_gene(head_length, tail_length, ...)
        tree = decode_gene(gene, head_length)
    return genome
```

#### 2.4 变异算子增加结构变异
**文件**: `src/evolution/mutation.py`

当前变异主要是点变异和常数微调。建议增加:
```python
def subtree_replacement(genome, cfg, rng, prob=0.1):
    """子树替换：随机选择一个节点，用新随机子树替换"""
    if rng.random() > prob:
        return genome
    tree = decode_gene(genome.potential_gene, genome.head_length)
    # 随机选择一个非叶节点
    # 用新随机子树替换
    # 重新编码为基因
    return new_genome
```

---

### Phase 3: Grid 探索优化（优先级：中）

**目标**: 提高特征空间覆盖率，避免聚集在少数 bin

#### 3.1 特征选择优化
**文件**: `src/evolution/features.py`

当前 3D 特征: `(entropy, islands, fft_amp)`

问题:
- `entropy` 值域窄（0.87-0.92），区分度低
- `fft_amp` 受采样影响大

建议改为更具区分度的特征:
```python
def get_3d_features(self) -> tuple:
    features = self.compute_features()
    # 方案 A: (entropy, survival_rate, angular_momentum_skew)
    # 方案 B: (speed_variance, nutrient_consume, energy_skew)
    # 方案 C: (islands, autocorr_lag10, density_laplacian_var)
    return (float(features[4]), float(features[12]), float(features[14]))
```

#### 3.2 自适应 Grid 分辨率
**文件**: `src/evolution/map_elites.py`

当前固定分辨率 5×5×5=125。建议根据填充率动态调整:
```python
def maybe_resize_grid(self):
    """当填充率 > 60% 时，增加分辨率"""
    fill_ratio = self.get_fill_ratio()
    if fill_ratio > 0.6 and self.resolution < 15:
        self.resolution += 2
        # 重新映射现有 grid entries 到新分辨率
        self._remap_grid()
```

#### 3.3 特征归一化改进
**文件**: `src/evolution/features.py`

当前特征没有归一化，导致某些特征主导距离计算。建议:
```python
def compute_features(self) -> np.ndarray:
    features = ...  # 现有计算
    # Min-max 归一化到 [0, 1]
    features = (features - self._feature_min) / (self._feature_max - self._feature_min + 1e-8)
    return features
```

---

### Phase 4: VLM 集成优化（优先级：中）

**目标**: 提高 VLM 评判质量和效率

#### 4.1 VLM 输入增强
**文件**: `src/vlm/judge.py`

当前 VLM 只看到压缩的轨迹图（128x128）。建议增强输入:
```python
def build_vlm_prompt(genome, features_15d, formula, env_summary):
    """构建更丰富的 VLM 输入"""
    prompt = f"""
    观察这个粒子系统涌现行为：

    公式: U = {formula}
    化趋性: {genome.to_formula_chemotaxis()}

    行为特征:
    - 空间熵: {features_15d[0]:.3f} (0=聚集, 1=均匀)
    - 岛屿数: {features_15d[2]:.0f} (连通分量数)
    - 速度方差: {features_15d[4]:.4f} (活动水平)
    - 存活率: {features_15d[10]:.2f}
    - 营养消耗: {features_15d[12]:.4f}
    - 能量偏度: {features_15d[14]:.4f}

    环境状态:
    - 营养总量: {env_summary['total_nutrient']:.2f}
    - 废弃总量: {env_summary['total_waste']:.2f}

    请描述你观察到的涌现行为模式。
    """
    return prompt
```

#### 4.2 VLM 缓存策略优化
**文件**: `src/vlm/judge.py`

当前缓存基于特征向量的精确匹配。建议改为模糊匹配:
```python
def should_call_vlm(features_12d, judged_features, novelty_score,
                    min_novelty_gap=0.1, use_cosine=True):
    """改进的 VLM 调用判断"""
    # 1. 高新颖性分数 → 直接调用
    if novelty_score >= 3.0:
        return True

    # 2. 余弦距离 < 0.1 → 跳过（太相似）
    # 3. 余弦距离 0.1-0.3 → 20% 概率调用（探索）
    # 4. 余弦距离 > 0.3 → 调用
    min_dist = compute_min_cosine_distance(features_12d, judged_features)
    if min_dist < 0.1:
        return False
    elif min_dist < 0.3:
        return rng.random() < 0.2
    else:
        return True
```

#### 4.3 VLM 批量评估
**文件**: `src/main.py`

当前每代最多 3 次 VLM 调用。建议批量处理:
```python
# 收集本代所有 VLM-worthy 样本
if novelty_packages:
    # 批量发送给 VLM（减少 I/O 开销）
    batch_response = vlm_judge_batch(novelty_packages, cfg)
    for pkg, response in zip(novelty_packages, batch_response):
        logger.info(f"VLM [{pkg['formula'][:40]}...]: {response[:100]}...")
```

---

### Phase 5: 性能优化（优先级：低）

**目标**: 提高每代运行速度

#### 5.1 采样间隔优化
**文件**: `config/fast_debug.yaml`

当前 `sample_interval: 100`，`steps_per_eval: 1000`，只采样 10 次。建议:
```yaml
novelty:
  sample_interval: 200  # 增加间隔，减少采样开销
  steps_per_eval: 2000  # 增加步数，提高特征稳定性
```

#### 5.2 GPU 内存优化
**文件**: `src/simulation/step.py`

当前每个粒子的力计算遍历所有邻居。建议:
- 使用更小的感知半径（当前 `cell_size * 1.5`）
- 减少 `vm_stack_depth`（当前 16，可以试 8）

#### 5.3 并行化优化
**文件**: `src/main.py`

当前 VLM 调用是串行的。建议:
- 使用线程池并行调用 VLM（如果 Ollama 支持）
- 或异步调用 VLM，不阻塞主循环

---

### Phase 6: 数据分析工具（优先级：低）

**目标**: 更好地理解进化过程

#### 6.1 实验对比工具
**文件**: `scripts/compare_experiments.py`（新增）

功能:
- 对比不同配置的进化曲线
- 可视化 Grid 填充率变化
- 公式多样性统计

#### 6.2 行为聚类分析
**文件**: `scripts/cluster_analysis.py`（新增）

功能:
- 对 archive 中的行为向量进行聚类
- 识别行为模式家族
- 生成行为图鉴

#### 6.3 交互式可视化
**文件**: `scripts/interactive_viewer.py`（新增）

功能:
- 浏览 Grid 中的公式和对应行为
- 播放粒子动画
- 查看环境场变化

---

## 实施优先级

| 阶段 | 任务 | 预估工时 | 依赖 |
|------|------|----------|------|
| Phase 1 | 环境可视化 | 2-3 天 | 无 |
| Phase 2 | 公式多样性 | 2-3 天 | 无 |
| Phase 3 | Grid 优化 | 1-2 天 | Phase 2 |
| Phase 4 | VLM 优化 | 2-3 天 | Phase 1 |
| Phase 5 | 性能优化 | 1-2 天 | 无 |
| Phase 6 | 分析工具 | 3-5 天 | Phase 1-4 |

**建议顺序**: Phase 1 → Phase 2 → Phase 4 → Phase 3 → Phase 5 → Phase 6

---

## 验证标准

### Phase 1 验证
- [ ] 截图中显示营养场/废弃场热力图
- [ ] VLM 判断提到环境场分布
- [ ] 粒子行为与环境场相关（如聚集在营养热点）

### Phase 2 验证
- [ ] Grid 中常数公式占比 < 20%
- [ ] 公式深度平均 > 3
- [ ] 适应度分布更均匀（非集中在 0.999）

### Phase 3 验证
- [ ] Grid 填充率 > 30%（60代后）
- [ ] 特征分布更均匀（各 bin 数量差异 < 3 倍）
- [ ] 新颖性分数方差增大

### Phase 4 验证
- [ ] VLM 判断准确描述行为模式
- [ ] VLM 缓存命中率 > 50%
- [ ] 无重复/相似的 VLM 判断

---

## 配置变更

### fast_debug.yaml 新增/修改

```yaml
# 环境可视化
rendering:
  show_nutrient: true
  show_waste: true
  show_gradients: true
  overlay_alpha: 0.3

# 公式多样性
evolution:
  min_formula_depth: 3
  complexity_bonus: 0.15
  constant_penalty: 0.5

# Grid 优化
map_elites:
  auto_resize: true
  resize_threshold: 0.6
  feature_normalize: true
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 环境可视化增加 GPU 内存 | 可能 OOM | 只在截图时提取到 CPU，不常驻 GPU |
| 公式复杂度奖励导致过拟合 | 适应度虚高 | 设置复杂度上限，监控泛化性 |
| Grid 自适应导致旧数据丢失 | 历史不可比 | 保留原始分辨率的备份 |
| VLM 批量调用增加延迟 | 每代时间增加 | 异步调用，不阻塞主循环 |

---

## 变更日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-06-25 | v1.0 | 初始版本，包含已修复的 4 个 hotfix |
