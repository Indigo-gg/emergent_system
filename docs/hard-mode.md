# Hard Mode 涌现系统：落地实施文档 v6

> 8GB VRAM 工作站 · 长期无人值守运行 · 百万级粒子 · 开放式进化探索

---

## 〇、设计哲学

> **弱控制，强规则，形式追随物理。规则的制定权不交给神经网络，而是交给进化本身。**

本方案的核心原则：
1. **可解释性优先**：基因组不是黑盒权重，而是人类可读的数学公式。
2. **动态 > 静态**：涌现的本质是行为，不是图案。评估基于时序不变量，而非原始时序或单帧截图。
3. **物理正确性**：GEP 生成标量势能场 U，力由**符号求导** `F = -dU/dr` 计算（非数值求导），天然满足动量守恒且无精度损失。
4. **编译一次，执行任意**：公式变化只改字节码数据，不改 kernel 代码——零重编译开销。
5. **数值安全**：五重防护（安全函数集 + 符号求导势能力 + 粘性阻尼 + clamp + NaN 检测）。
6. **只存公式，不存轨迹**：用相同种子重跑可复现**宏观形态**（非像素级精确），存储节省 400 万倍。
7. **成本可控**：本地 VLM（Qwen2.5-VL-3B via Ollama）CPU 模式运行，零 API 费用，不与仿真争抢显存。
8. **环境驱动涌现**：粒子在空旷世界里只靠力交互，涌现空间有限。引入**营养场 + 废弃物场**作为环境层，粒子需要争夺资源、避开毒素，适应度由环境自然选择而非人工定义 fitness 函数。环境反馈是涌现的放大器。

---

## 一、目标定义

**一句话**：用 GPU 跑百万级粒子在**营养-废弃物环境**中的开放式进化，进化产出人类可读的物理规则公式，粒子需要争夺资源、适应环境变化，用 MAP-Elites 收集行为多样性，用时序特征过滤无聊数据，只把最惊艳的 1% 送给 VLM 评判。

**交付物**：
- 一个可长期无人值守运行的 Python 程序
- 一本"涌现生物图鉴"：MAP-Elites 地图 + 每个物种的规则公式 + 行为描述
- 一套可回放、可追溯的实验数据档案

---

## 二、技术栈

| 层级 | 选型 | 理由 |
|------|------|------|
| 计算框架 | **Taichi** | 原生支持 GPU 动态数据结构（Sparse Grid、原子操作），Python 语法写 GPU Kernel，空间哈希无需 padding |
| 公式执行 | **Taichi 栈式虚拟机** | 预编译一次 kernel，公式变化只改字节码数据，零重编译开销 |
| 基因表达 | **GEP（基因表达式编程）** | 进化产出人类可读的数学公式，保留"意图-裁决"的可解释性 |
| 行为编码 | **时序特征** | 基于宏观统计量的时间序列（非单帧截图），捕捉动态涌现 |
| 特征档案 | **3D MAP-Elites + Novelty Archive** | 精英网格密集竞争 + 开放式新颖性归档，避免维度灾难 |
| VLM 评判 | **Qwen2.5-VL-3B（Ollama）** | Ollama 管理模型生命周期，HTTP API 调用。CPU 模式运行避免与 Taichi 争抢显存。零 API 费用 |
| 配置管理 | **YAML + dataclass** | 参数可版本化，实验可复现 |
| 日志监控 | **Python logging** | 长期运行必须有日志 |

---

## 三、项目结构

```
emergent-system-hard/
├── docs/
│   └── hard-mode.md              # 本文档
├── config/
│   ├── default.yaml              # 默认实验参数
│   └── profiles/
│       ├── fast_debug.yaml
│       └── full_scale.yaml
├── src/
│   ├── main.py                   # 入口：解析配置、启动演化循环
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── particles.py          # 粒子状态（Taichi fields）
│   │   ├── potential.py          # 势能场计算（GEP 字节码 → U，符号求导 → dU/dr）
│   │   ├── spatial_hash.py       # GPU 原生空间哈希（Taichi sparse struct）
│   │   └── step.py               # 单步推进（Taichi kernel）
│   ├── evolution/
│   │   ├── __init__.py
│   │   ├── genome.py             # GEP 基因组：表达式树 + 遗传算子
│   │   ├── gep.py                # GEP 引擎：编码/解码/适应度
│   │   ├── mutation.py           # 变异算子（点变异、转座、重组）
│   │   ├── map_elites.py         # MAP-Elites 档案管理（5-6 维）
│   │   └── features.py           # 15 维时序不变量提取（含环境特征）
│   ├── rendering/
│   │   ├── __init__.py
│   │   └── renderer.py           # GPU 渲染：快照 + 轨迹图 + GIF
│   ├── novelty/
│   │   ├── __init__.py
│   │   └── filter.py             # 新颖性判定（k-NN 基于行为轨迹）
│   ├── vlm/
│   │   ├── __init__.py
│   │   └── judge.py              # VLM 调用：轨迹图 + 时序摘要 + 公式
│   └── storage/
│       ├── __init__.py
│       ├── db.py                 # SQLite 读写
│       └── archive.py            # 文件落盘管理
├── data/
│   ├── experiments/
│   │   └── {experiment_id}/
│   │       ├── config.yaml
│   │       ├── map_elites.db
│   │       ├── formulas/         # GEP 公式存档（.txt，人类可读）
│   │       ├── trajectories/     # 行为轨迹数据（.npz）
│   │       ├── screenshots/      # 新颖性告警截图
│   │       └── logs/
├── scripts/
│   ├── vm_pressure_test.py      # Phase 1 最高优先级：VM 性能验证
│   └── visualize_map.py
├── tests/
└── requirements.txt
```

---

## 四、核心模块设计

### 4.1 粒子仿真 (`simulation/`)

**状态表示**：每个粒子 `[x, y, vx, vy, energy, internal_state(4维)]`
- `energy`：粒子能量，由营养场补充，被代谢消耗，归零则粒子休眠/死亡
- `internal_state`（4维）：由 GEP 字节码驱动的可塑内部状态

**单步流程**（Taichi kernel，全在 GPU）：
1. **构建空间哈希**（含桶容量硬上限）
2. **邻居查询**：遍历 3×3 邻域格子，每个桶内链表遍历
3. **势能力计算**：通过栈式 VM 执行 GEP 字节码计算势能 U，力由 F = -∇U 获得。GEP 终端集包含 `nutrient` 和 `waste` 变量（当前粒子位置的场值），公式可进化出环境感知能力
4. **粘性阻尼**：`F_total = F_potential - γ * v`（防止数值振荡）
5. **状态更新**：内部状态由另一个 GEP 字节码驱动
6. **环境交互**：
   - 摄取营养：`energy += nutrient[x,y] * absorb_rate`，同时消耗该位置营养
   - 代谢消耗：`energy -= base_metabolism + |v| * move_cost`
   - 废弃物沉积：聚集区域产生废弃物 `waste[x,y] += waste_rate`
   - 废弃物惩罚：高废弃物区域能量消耗倍增 `energy -= waste[x,y] * waste_penalty`
7. **生命判定**：`energy <= 0` → 进入休眠状态（极低代谢，附近有营养时苏醒）；休眠超过阈值 → 正式死亡
8. **积分 + 速度/位移限制 + 边界处理**

#### 环境层：营养场 + 废弃物场

> **核心思想**：粒子在空旷世界里只靠力交互，涌现空间有限。引入环境层后，粒子需要争夺资源、避开毒素，适应度由环境自然选择而非人工定义 fitness 函数。这是涌现的放大器。

**环境场表示**（Taichi sparse grid，与粒子场同分辨率）：
```python
nutrient_field   # 营养浓度场，float32
waste_field      # 废弃物浓度场，float32
```

**环境动力学**（每步执行）：
```python
@ti.kernel
def environment_step():
    # 1. 营养扩散 + 衰减
    for i, j in nutrient_field:
        laplacian = (nutrient_field[i+1,j] + nutrient_field[i-1,j] +
                     nutrient_field[i,j+1] + nutrient_field[i,j-1] -
                     4 * nutrient_field[i,j])
        nutrient_field[i,j] += diffuse_rate * laplacian - decay_rate * nutrient_field[i,j]
        nutrient_field[i,j] = max(nutrient_field[i,j], 0.0)

    # 2. 废弃物扩散 + 衰减
    for i, j in waste_field:
        laplacian = (waste_field[i+1,j] + waste_field[i-1,j] +
                     waste_field[i,j+1] + waste_field[i,j-1] -
                     4 * waste_field[i,j])
        waste_field[i,j] += waste_diffuse * laplacian - waste_decay * waste_field[i,j]
        waste_field[i,j] = max(waste_field[i,j], 0.0)

    # 3. 营养注入（移动热点）
    for patch in nutrient_patches:
        cx, cy = patch.center  # 沿螺旋/随机路径漂移
        for (i,j) near (cx, cy):
            nutrient_field[i,j] += patch.amount * gaussian(dist(i,j,cx,cy))
        patch.angle += drift_speed  # 热点漂移
```

**为什么用移动热点？**
- 固定营养源 → 粒子找到后静止不动 → 静态平衡 → 无涌现
- 移动热点 → 粒子必须追逐、迁徙、争夺 → 动态涌现（领地、迁徙路线、觅食策略）
- 营养漂移路径可参数化（螺旋、随机游走、正弦波），不同路径产生不同涌现模式

**反灭绝机制**：
- 粒子能量归零 → 进入**休眠孢子**状态（不移动、极低代谢 0.001/tick）
- 附近有营养时苏醒
- 最大休眠时间 600 tick，超时则真正死亡
- 作用：防止整个种群在环境恶劣时瞬间灭绝，给进化"第二次机会"

**废弃物的涌现效果**：
- 聚集 → 过度消耗营养 → 废弃物爆发 → 环形逃离 → 脉冲式扩张收缩
- 类似 B-Z 化学振荡（Belousov-Zhabotinsky reaction）
- 不需要 GEP 公式编码振荡行为，环境反馈自动产生

**环境对 GEP 进化的影响**：
- GEP 终端集新增 `nutrient`（当前位置营养浓度）和 `waste`（当前位置废弃物浓度）
- 进化可以产出：
  - 趋营养公式：`U = -k * nutrient * dist`（营养越多，吸引力越大）
  - 避废弃物公式：`U = k * waste / (dist + eps)`（废弃物越多，排斥力越大）
  - 复合策略：`U = sin(nutrient * 3.14) - 0.5 * waste * dist`
- 这些公式是**环境感知**的，比纯力交互丰富得多

#### 空间哈希：桶容量硬上限

> **问题**：强引力公式会让 100 万粒子坍缩到同一个桶 → O(N²) → GPU TDR 崩溃。
>
> **解决**：每个桶最多容纳 `BUCKET_MAX = 128` 个粒子。超出部分拒绝插入。

```python
BUCKET_MAX = 128

@ti.kernel
def build_hash():
    for i in range(n_particles):
        cell = int(pos[i] / cell_size)
        # 原子计数：检查桶是否已满
        count = ti.atomic_add(cell_count[cell], 1)
        if count < BUCKET_MAX:
            old = cell_head[cell]
            cell_head[cell] = i
            particle_next[i] = old
        else:
            particle_next[i] = -1  # 拒绝插入，该粒子本步不参与邻居交互
```

#### 动量守恒：GEP 生成势能场 + 符号求导

> **问题**：GEP 随机生成的力公式 F_{ij} = f(dist, density) 通常不对称 → F_{ij} ≠ -F_{ji} → 动量不守恒 → 粒子团凭空获得整体动量 → 全屏宏观平移。
>
> **解决**：GEP 不生成力，而是生成**标量势能场 U**。力由**符号求导** `F = -dU/dr` 计算。

**为什么不求数值求导？**
- `float32` 下 `U(r+eps) - U(r)` 当 U 很大时，有效数字被截断（Catastrophic Cancellation）
- 产生"幻觉力"，导致粒子行为与公式意图完全不符
- 符号求导零精度损失，还省一次 VM 执行开销

**符号求导实现**：
```python
# CPU 端：对 GEP 表达式树进行符号求导
def symbolic_diff(tree: Node, var_name: str) -> Node:
    """
    对表达式树关于 var_name 求偏导，返回新的表达式树。
    链式法则：d/dx[f(g(x))] = f'(g(x)) * g'(x)
    """
    if tree.is_leaf():
        if tree.name == var_name:
            return ConstNode(1.0)  # d(dist)/d(dist) = 1
        else:
            return ConstNode(0.0)  # d(density)/d(dist) = 0
    elif tree.op == 'add':
        return Add(symbolic_diff(tree.left, var_name), symbolic_diff(tree.right, var_name))
    elif tree.op == 'mul':
        # 乘法法则：d(f*g)/dx = f'*g + f*g'
        return Add(
            Mul(symbolic_diff(tree.left, var_name), tree.right),
            Mul(tree.left, symbolic_diff(tree.right, var_name))
        )
    elif tree.op == 'sin':
        # d(sin(f))/dx = cos(f) * f'
        return Mul(Cos(tree.child), symbolic_diff(tree.child, var_name))
    elif tree.op == 'cos':
        # d(cos(f))/dx = -sin(f) * f'
        return Mul(Neg(Sin(tree.child)), symbolic_diff(tree.child, var_name))
    # ... 其余函数同理
```

**编译流程**：
```
GEP 生成势能树 U(dist, density, ...)
        ↓ CPU 端符号求导
生成 dU/dr 的表达式树
        ↓ CPU 端编译为字节码
生成 dU/dr 的字节码（传入 GPU）
        ↓ GPU 端执行
force = -dU/dr 的 VM 执行结果
```

**效果**：粒子 i 对 j 的力 = -(dU/dr)_{ij}，粒子 j 对 i 的力 = -(dU/dr)_{ji}，在同一距离下大小相等方向相反 → 动量守恒。且无浮点精度损失。

#### 粘性阻尼与速度限制

> **问题**：OP_CLAMP 截断力 → 粒子在两点间高频跨越式振荡 → FFT 假峰值 → 进化出"数值 Bug 物种"。
>
> **解决**：粘性阻尼 + 单步最大位移限制。

```python
gamma = 0.1  # 阻尼系数

@ti.func
def integrate(i: int, dt: ti.f32):
    # 粘性阻尼：F_total = F_potential - γ * v
    fx = force_x[i] - gamma * vel_x[i]
    fy = force_y[i] - gamma * vel_y[i]

    # 速度更新
    vel_x[i] += fx * dt
    vel_y[i] += fy * dt

    # 速度限制
    speed = ti.sqrt(vel_x[i]**2 + vel_y[i]**2)
    max_speed = 5.0
    if speed > max_speed:
        vel_x[i] *= max_speed / speed
        vel_y[i] *= max_speed / speed

    # 单步最大位移限制：v*dt < 0.5 * cell_size
    max_disp = 0.5 * cell_size
    dx = vel_x[i] * dt
    dy = vel_y[i] * dt
    disp = ti.sqrt(dx**2 + dy**2)
    if disp > max_disp:
        dx *= max_disp / disp
        dy *= max_disp / disp

    pos_x[i] += dx
    pos_y[i] += dy
```

#### 栈式虚拟机（VM）

> **问题**：每代 10 个新公式 × 动态编译 kernel = GPU 99% 时间在等 LLVM 编译。
>
> **解决**：预编译一个 Taichi kernel，内部包含极小的栈式 VM。公式序列化为定长字节数组，作为数据传入 GPU。

```python
OP_CONST = 0; OP_VAR = 1
OP_ADD = 10; OP_SUB = 11; OP_MUL = 12
OP_SIN = 20; OP_COS = 21; OP_TANH = 22; OP_SQRT = 23; OP_ABS = 24
OP_MAX = 30; OP_MIN = 31
OP_CLAMP = 40; OP_HALT = 255

@ti.func
def vm_execute(bytecode, constants, vars) -> ti.f32:
    stack = ti.Vector([0.0] * 16)
    sp = 0; pc = 0
    while True:
        op = bytecode[pc]
        if op == OP_CONST:
            sp += 1; stack[sp] = constants[bytecode[pc+1]]; pc += 2
        elif op == OP_VAR:
            sp += 1; stack[sp] = vars[bytecode[pc+1]]; pc += 2
        elif op == OP_ADD:
            stack[sp-1] = stack[sp-1] + stack[sp]; sp -= 1; pc += 1
        elif op == OP_MUL:
            stack[sp-1] = stack[sp-1] * stack[sp]; sp -= 1; pc += 1
        elif op == OP_SIN:
            stack[sp] = ti.sin(stack[sp]); pc += 1
        elif op == OP_CLAMP:
            stack[sp] = ti.math.clamp(stack[sp], -100.0, 100.0); pc += 1
        elif op == OP_HALT:
            break
        # ... 其余指令同理
    return stack[sp]
```

**VM 性能风险与应对**：

| 风险 | 硬件原因 | 应对 |
|------|----------|------|
| **寄存器溢出（Register Spilling）** | `stack = ti.Vector([0.0]*16)` 若用动态索引访问，LLVM 无法放入寄存器，溢出到本地内存（延迟 100x） | Phase 1 必须用 Taichi profiler 检查 kernel，确认 `stack` 被寄存器化。若溢出，改为基于寄存器的 VM（固定 4 个寄存器 + spill 到栈）或 SIMD 化 VM |
| **分支发散（Warp Divergence）** | 不同粒子执行不同字节码路径 → warp 内线程串行化 | 所有粒子执行同一字节码（全局统一公式），分支仅来自字节码指令类型，可接受 |
| **SIMD 化 VM（备选方案）** | 如果栈式 VM 太慢 | 外层循环遍历指令，内层循环遍历所有粒子的邻居对。保证 GPU 内存访问连续性 |

**Phase 1 压力测试清单**：
```bash
# 必须在写主逻辑之前完成
python scripts/vm_pressure_test.py
# 测试项：
# 1. 100 万粒子 × 100 步 VM 执行 → 目标 < 500ms
# 2. Taichi profiler 检查 stack 是否寄存器化
# 3. 对比硬编码公式 vs VM 执行的性能比 → 目标 < 5x
```

**GEP 代码膨胀（Bloat）消除**：
- CPU 端编译字节码时，执行**死代码消除**：标记每个节点是否被最终结果依赖，删除无用子树
- 适应度中对**实际有效指令数**（非数组长度）施加 parsimony pressure

### 4.2 基因组：GEP 基因表达式编程 (`evolution/gep.py`)

**为什么不用 MLP 权重？**

MLP 权重是黑盒。看到一个惊艳的"黏菌迷宫"涌现时，你只能说"这是第 1420 代第 3 个基因组的第二层权重碰巧演化成了这样"。

GEP 进化产出的是**标量势能场公式**：`U = 1.5 * dist - sin(density * 3.14)`。力由底层统一计算 `F = -∇U`，天然满足动量守恒。

v6 新增：GEP 终端集包含 `nutrient` 和 `waste` 变量，公式可进化出环境感知能力：
- 趋营养：`U = -2.0 * nutrient * dist`
- 避废弃物：`U = 1.5 * waste / (dist + 0.1)`
- 复合策略：`U = sin(nutrient * 3.14) - 0.5 * waste * dist`

**GEP 基因组结构**：
```python
@dataclass
class GEPGenome:
    # 势能场规则：标量势能 U（力由 F = -∇U 自动获得，保证动量守恒）
    # 终端变量：dist, density, speed, angle, state[0..3], neighbor_count, nutrient, waste
    potential_expr: str      # 例如: "1.5 * dist - sin(density * 3.14) * 0.5 - 2.0 * nutrient"
    # 状态更新规则：内部状态的演化
    state_expr: str          # 例如: "tanh(state[0] + avg_neighbor_state * 0.3)"
    # 感知规则：感知范围和权重
    sense_expr: str          # 例如: "2.0 + 0.5 * speed"
    # 代谢规则：能量消耗策略（由环境选择压力驱动进化）
    metabolism_expr: str     # 例如: "0.01 + 0.005 * speed + 0.02 * waste"
    # 元数据
    metadata: dict           # 代数、来源、变异历史、随机种子
```

**GEP 编码方式**：
- 基因 = 一棵表达式树，序列化为字节码（供 GPU 栈式 VM 执行）
- 终端集（Terminal Set）：`{dist, density, speed, angle, state[0..3], neighbor_count, nutrient, waste}`（输入到势能函数的变量。`nutrient` = 当前粒子位置的营养浓度，`waste` = 当前粒子位置的废弃物浓度）
- 函数集（Function Set，安全子集）：`{+, -, *, sin, cos, tanh, sqrt, abs, max, min}`
- 常数集（Constant Set）：`[-5.0, 5.0]` 内的浮点数，可被变异

**函数集的安全设计**：
- **排除 `/`（除法）**：除以零 → NaN/Inf，改为 `a * (1/b)` 且 `b` 先 clamp 到 `[0.01, 100]`
- **排除 `exp`**：指数爆炸是 NaN 瘟疫的主要来源，`exp(100)` = `2.7e43` 直接溢出
- **排除 `pow`**：`pow(0, -1)` = Inf，`pow(-2, 0.5)` = NaN
- **排除 `log`**：`log(0)` = `-Inf`
- 保留 `sin/cos/tanh`（天然有界，且符号求导简单）和 `sqrt/abs`（仅需 clamp 输入 ≥ 0）

**符号求导支持的函数**：
| 函数 | 导数 | 求导复杂度 |
|------|------|-----------|
| `f + g` | `f' + g'` | O(1) |
| `f * g` | `f'*g + f*g'` | O(1) |
| `f - g` | `f' - g'` | O(1) |
| `sin(f)` | `cos(f) * f'` | O(1) |
| `cos(f)` | `-sin(f) * f'` | O(1) |
| `tanh(f)` | `(1 - tanh²(f)) * f'` | O(1) |
| `sqrt(f)` | `f' / (2*sqrt(f))` | O(1) |
| `abs(f)` | `sign(f) * f'` | O(1) |
| `max(f, g)` | `f' if f > g else g'` | O(1) |
| `min(f, g)` | `f' if f < g else g'` | O(1) |

求导后的表达式树大小约为原树的 2-3x，通过 `simplify()` + `eliminate_dead_code()` 控制字节码长度。

**字节码编译**（含符号求导）：
```python
def gep_to_bytecodes(tree: Node) -> tuple[list[int], list[int], list[float]]:
    """
    将 GEP 势能树编译为两套字节码：U 的字节码 + dU/dr 的字节码。
    dU/dr 通过符号求导生成（非数值求导），零精度损失。
    """
    # 1. 编译势能 U 的字节码
    u_bytecode, constants = _compile(tree)
    u_bytecode.extend([OP_CLAMP, OP_HALT])

    # 2. 符号求导生成 dU/dr 的表达式树
    dudr_tree = symbolic_diff(tree, var_name='dist')
    # 3. 简化求导结果（消除 *0, +0, *1 等冗余节点）
    dudr_tree = simplify(dudr_tree)
    # 4. 死代码消除
    dudr_tree = eliminate_dead_code(dudr_tree)
    # 5. 编译 dU/dr 的字节码
    dudr_bytecode, _ = _compile(dudr_tree, constants)  # 共享常量池
    dudr_bytecode.extend([OP_CLAMP, OP_HALT])

    return u_bytecode, dudr_bytecode, constants
```

**关键**：GPU 端执行时，直接执行 `dU/dr` 的字节码得到力，不需要再执行 U 的字节码 + 数值求导。省一半 VM 开销。

**遗传算子**：
| 算子 | 概率 | 作用 |
|------|------|------|
| 点变异 | 30% | 随机改变树中一个节点（函数/终端/常数） |
| 常数微调 | 20% | 对常数加高斯噪声 |
| 转座（IS） | 15% | 将子树插入另一位置 |
| 转座（RIS） | 10% | 将子树插入根节点 |
| 单点重组 | 15% | 两个亲本交换子树 |
| 双点重组 | 10% | 两个亲本交换两段子树 |

**适应度函数**（多目标加权）：

v6 改进：有了营养/废弃物环境后，适应度可以更多地由**环境自然选择**驱动，而非人工定义指标：

```
# 环境驱动的适应度（主成分）
fitness_env = w1 * total_energy_harvested     # 粒子从环境中获取的总能量
            + w2 * survival_rate              # 存活率（环境选择的直接结果）
            + w3 * energy_variance            # 能量方差（鼓励多样化策略）

# 结构复杂度（辅助成分）
fitness_struct = w4 * structure_complexity     # 空间结构的熵

# 效率指标（辅助成分）
fitness_eff = w5 * energy_efficiency           # 能量利用效率

fitness = fitness_env + fitness_struct + fitness_eff

# NaN/Inf 个体直接赋 0 分，不浪费仿真时间
if any_nan_or_inf(fitness):
    fitness = 0.0
```

**为什么环境驱动比人工定义更好？**
- 人工定义 fitness 需要人类猜测"什么是好的涌现"，容易遗漏
- 环境驱动的 fitness 让"能活下来的就是好的"——自然选择不需要人类预设目标
- 营养竞争 + 废弃物惩罚 自动产生选择压力，进化出的公式必须同时解决"找营养"和"避毒素"两个问题

### 4.3 特征提取：时序不变量 (`evolution/features.py`)

> **问题**：直接拼接原始时间序列（500 维）计算 k-NN 距离会失效。物种 A 在第 0 步脉冲，物种 B 在第 50 步脉冲 → 欧氏距离巨大 → 误判为不同物种。
>
> **解决**：用**时序不变量（Time-invariant Features）**替代原始时序。将动态时序压缩为统计矩和频域特征，消除相位对齐问题。

**特征维度（15 维，全部是时序不变量）**：

| # | 特征 | 计算方式 | 捕捉什么 |
|---|------|----------|----------|
| 1 | **空间熵均值** | 整个仿真期间空间熵的均值 | 有序 vs 混乱 |
| 2 | **空间熵方差** | 空间熵的方差 | 稳定 vs 波动 |
| 3 | **岛屿数均值** | 岛屿数量的均值 | 聚合 vs 分散 |
| 4 | **岛屿数方差** | 岛屿数量的方差 | 稳定 vs 分裂-融合 |
| 5 | **速度方差均值** | 速度方差的均值 | 活跃度 |
| 6 | **速度方差 FFT 前3主频幅值（3维）** | 速度方差时序 → FFT → 取前3峰值的幅值 | 振荡模式 |
| 7 | **角动量偏度** | 角动量分布的偏度（skewness） | 旋转 vs 平移 |
| 8 | **密度拉普拉斯方差均值** | ∇²ρ 的方差的均值 | 均匀 vs 纹理化 |
| 9 | **存活率** | 仿真结束时存活粒子比例 | 自维持 vs 衰亡 |
| 10 | **速度方差自相关系数** | 速度方差时序的 lag-10 自相关 | 周期性 vs 随机性 |
| 11 | **营养消耗率均值** | 每步营养消耗量的均值 | 觅食能力 |
| 12 | **废弃物峰值均值** | 废弃物场最大值的均值 | 聚集副作用 |
| 13 | **能量分布偏度** | 粒子能量分布的偏度 | 资源分配不均 vs 均衡 |

v6 新增 3 维环境特征（11-13），反映粒子与环境的交互模式。

**为什么用不变量而非原始时序？**
- 原始时序受相位影响：相同行为、不同起始相位 → 欧氏距离大 → 误判为不同物种
- 统计量（均值、方差、偏度、FFT 幅值、自相关）对相位不敏感
- 15 维 vs 500 维：计算量降低 33x，且更鲁棒

**行为向量（用于 Novelty Archive 的 k-NN）**：
```python
behavior_vector = [
    entropy_mean, entropy_var,
    islands_mean, islands_var,
    speed_var_mean,
    fft_amp_1, fft_amp_2, fft_amp_3,  # FFT 前 3 主频幅值
    angular_skew,
    density_lap_var_mean,
    survival_rate,
    autocorr_lag10,
    nutrient_consume_mean,  # v6: 营养消耗率
    waste_peak_mean,        # v6: 废弃物峰值
    energy_skew,            # v6: 能量分布偏度
]
# 15 维，直接计算 k-NN 距离
```

**3D MAP-Elites 网格特征**（取最有区分度的 3 个）：
- X 轴：空间熵均值（有序 vs 混乱）
- Y 轴：岛屿数均值（聚合 vs 分散）
- Z 轴：速度方差 FFT 第一主频幅值（振荡 vs 稳态）

**死寂宇宙过滤（Minimum Viability Test）**：
- 如果 `survival_rate < 0.1`（90% 粒子死亡）→ fitness = 0，不进入任何档案
- 如果 `entropy_mean > 0.95 * max_entropy`（完全均匀分布）→ fitness = 0
- 如果 `speed_var_mean < 0.001`（所有粒子静止）→ fitness = 0
- v6 新增：如果 `nutrient_consume_mean < 0.001`（粒子完全不进食）→ fitness = 0
- v6 新增：如果 `energy_skew > 3.0`（极少数粒子垄断所有能量）→ fitness = 0
- 只有"处于混沌边缘"的数据才有资格被评估

### 4.4 特征档案：混合架构 (`evolution/map_elites.py`)

> **问题**：6 维 × 10 格 = 100 万格子，每代 10 个样本。10000 代才能填 10%。格子太稀疏 → MAP-Elites 选择亲本时变成盲目游走 → 进化收敛性极差。
>
> **解决方案**：**混合架构** — 3D 精英网格 + 无边界 Novelty Archive。

**架构图**：
```
┌─────────────────────────────────────────────┐
│              混合特征档案系统                  │
│                                               │
│  ┌─────────────────────┐  ┌────────────────┐ │
│  │ 3D MAP-Elites 网格  │  │ Novelty Archive │ │
│  │                      │  │                 │ │
│  │ 维度: 熵 × 岛屿 ×   │  │ 无格子限制      │ │
│  │       FFT主频        │  │ 基于距离的      │ │
│  │ 分辨率: 15×15×15     │  │ 开放式归档      │ │
│  │ 格子数: 3,375        │  │                 │ │
│  │                      │  │ 存储所有        │ │
│  │ 作用: 密集竞争       │  │ "首次见到"的    │ │
│  │ 保证局部最优         │  │ 行为模式        │ │
│  └──────────┬──────────┘  └───────┬────────┘ │
│             │                      │          │
│             ▼                      ▼          │
│        选择亲本时：          新颖性判定时：     │
│        从网格中采样          与 archive 比较    │
└─────────────────────────────────────────────┘
```

**3D MAP-Elites 网格**（密集竞争）：
- 3 个最有区分度的时序不变量：空间熵均值 × 岛屿数均值 × FFT 第一主频幅值
- 每轴 15 格 = 3,375 个格子
- 10000 代 × 10 样本/代 = 100,000 样本 → 填充率 ~100%（多次竞争同一格子）
- 保证：同一格子内的个体是真正的"局部最优"

**Novelty Archive**（开放式探索）：
- 无格子限制，存储所有被判定为"新颖"的行为模式
- 新颖性判定：当前行为向量与 archive 中所有向量的 k-最近邻平均距离
- k = 15（近邻数），阈值自适应
- 作用：防止系统只在 3D 网格内精耕，鼓励发现全新的行为类别

**亲本选择策略**：
```python
def select_parent():
    if random() < 0.7:
        # 70% 从 MAP-Elites 网格选（偏好高质量）
        return grid.random_non_empty_cell().genome
    else:
        # 30% 从 Novelty Archive 选（偏好多样性）
        return archive.random_entry().genome
```

**数据模型**：
```
MAP-Elites Grid cell (i, j, k) → {
    genome: GEPGenome,           # 含 potential_expr（势能公式）
    formula_text: str,           # "U = 1.5 * dist - sin(density) - 2.0 * nutrient"
    fitness: float,
    features_3d: (entropy_mean, islands_mean, fft_amp_1),
    features_15d: ndarray(15),   # 完整时序不变量（含环境特征）
    random_seed: int,            # 用于复现
    screenshot_path: str,
    generation: int,
    vlm_judgment: str | None,
}

Novelty Archive entry → {
    genome: GEPGenome,
    formula_text: str,
    behavior_vector: ndarray(15),  # 15 维时序不变量（含环境特征）
    novelty_score: float,
    features_15d: ndarray(15),
    random_seed: int,
    generation: int,
}
```

**核心操作**：
- `evaluate(genome) → (features_3d, features_6d, fitness, behavior_vector)`
- `grid.archive(genome, features_3d, fitness)` — 放入 3D 网格，保留更优者
- `novelty_archive.try_add(genome, behavior_vector)` — 如果新颖则加入 archive
- `select_parent() → GEPGenome` — 混合选择（网格 70% + archive 30%）
- `export_all() → str` — 导出网格公式 + archive 公式

**运行模式**：
- 每代：采样 10 个基因组 → 并行仿真 → 提取特征 → 归档到网格 + 尝试加入 archive
- 每 100 代：导出地图快照 + 公式集 + archive 统计
- 每 1000 代：保存完整 checkpoint

### 4.5 新颖性判定 (`novelty/filter.py`)

行为向量使用 4.3 节定义的 **15 维时序不变量**（均值、方差、偏度、FFT 幅值、自相关 + 环境特征），对相位不敏感。

**新颖性判定**（基于 Novelty Archive 的 k-NN）：
```python
def novelty_score(behavior_vec, archive_vectors, k=15):
    """
    behavior_vec: 15 维时序不变量向量（含环境特征）
    archive_vectors: archive 中所有已见向量
    novelty = 与 k 个最近邻的平均距离
    """
    if len(archive_vectors) < k:
        return float('inf')
    distances = np.linalg.norm(archive_vectors - behavior_vec, axis=1)
    k_nearest = np.sort(distances)[:k]
    return np.mean(k_nearest)
```

**自适应阈值**：
- 初期阈值低（archive 小，鼓励探索）
- 随着 archive 增长，阈值自动提高（取 archive 内所有 novelty_score 的中位数）
- 连续 10 代无新颖发现 → 阈值降低 20%

### 4.6 渲染 (`rendering/`)

**快照**（每代每个基因组）：粒子密度热力图，用于 MAP-Elites / Archive 档案

**轨迹图**（仅新颖性告警时）：
- 用存储的随机种子重跑仿真，生成最近 500 帧的轨迹叠加（颜色编码时间）
- 暴露"正在坍缩" vs "正在扩散" vs "稳定振荡"等动态行为

**时序特征曲线**（仅新颖性告警时）：
- 速度方差、熵、角动量的时间序列折线图
- 与轨迹图一起发给 VLM

**GIF**（仅新颖性告警时）：
- 用种子重跑，生成最近 200 帧动画，10fps
- 直接发给 VLM，让它看到"动"的涌现

### 4.7 VLM 评判 (`vlm/`)

**触发条件**：新颖性过滤器判定为 novel

**VLM 运行方式：Ollama（推荐）**

> 使用 Ollama 运行本地视觉模型。Ollama 独立管理模型生命周期，通过 HTTP API 调用，无需在 Python 中加载 PyTorch 模型。

**Ollama 优势**：
- 模型常驻内存/Ollama 自动管理显存，无需手动加载/卸载
- HTTP API 调用简单（`POST /api/generate`），Python 端只需 `requests` 库
- 支持多种视觉模型：`llava:7b`、`llava:13b`、`minicpm-v` 等
- 如果 Ollama 和 Taichi 争抢 8GB 显存，可将 Ollama 跑在 CPU 模式（`OLLAMA_NUM_GPU=0`）或用更小的模型

**推荐模型**：
| Ollama 模型 | 大小 | 说明 |
|------------|------|------|
| `llava:7b` | ~4GB | LLaVA 7B，通用视觉理解 |
| `minicpm-v` | ~5GB | MiniCPM-V，中文优化，图表理解强 |
| `llava:13b` | ~8GB | 更强但显存紧张 |

**调用方式**：
```python
import requests
import base64

def vlm_judge(image_path: str, prompt: str, model: str = "llava:7b") -> str:
    """Call Ollama VLM API."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
    })
    return resp.json()["response"]
```

**显存协调**（Ollama + Taichi 共存）：
- **推荐方案**：Ollama 用 CPU 模式（`OLLAMA_NUM_GPU=0`），慢但完全不争 GPU 显存。每天 50-100 次调用，每次 30-60 秒，总开销 ~30-60 分钟/天，对无人值守运行可接受。
- 备选方案 B：暂停仿真 → ti.reset() → 调用 Ollama → ti.init(arch=ti.cuda) → 恢复（需要代码实现显存协调逻辑）
- 备选方案 C：用更小的模型（`moondream` 约 1.7GB，加上 Taichi 的 5GB 可能刚好够）

**频率**：每天最多触发 ~50-100 次 VLM（新颖性告警），每次 ~10-30 秒，总开销 ~15-30 分钟/天。

**VLM 输入包**（3 件套）：
1. **轨迹图**（PNG）：最近 500 帧叠加，颜色编码时间（用随机种子重跑生成）
2. **时序特征曲线**（PNG）：速度方差、熵、角动量的时间序列折线图
3. **文本摘要**：
```
发现了一个新颖涌现形态。

势能公式：
  U = 1.5 * dist - sin(density * 3.14) * 0.5
  力由 F = -∇U 计算，保证动量守恒

状态更新：
  state' = tanh(state[0] + avg_neighbor_state * 0.3)

时序不变量特征：
  空间熵均值: 3.42 (历史均值: 2.1)
  岛屿数均值: 7 (历史均值: 3)
  FFT 第一主频幅值: 0.08 (存在振荡)
  角动量偏度: 1.3 (存在旋转结构)
  粒子存活率: 94% (自维持)
  自相关系数: 0.72 (周期性行为)

这是第 847 代，从第 812 代的 "U = 2.0 * dist - cos(density)" 变异而来。
复现种子: 42（用相同种子重跑可复现此形态）
```

**Prompt 模板**：
```
你是一个复杂系统研究员。以下是粒子仿真中发现的一个罕见涌现形态。

[轨迹图]
[时序特征曲线]

{文本摘要}

请：
1. 用一个简短的名字命名这个形态（2-4 个英文词）
2. 描述其动态行为特征（不是静态外观，而是"在做什么"）
3. 评估其新颖性（1-5 分，5 = 从未见过的行为模式）
4. 分析其势能公式的物理含义（为什么这个 U 会产生这种行为？力 F = -∇U 的效果是什么？）
5. 推测其在自然界中的类比（类似什么物理/生物系统？）
```

**与旧方案的关键区别**：
- 输入从"单帧截图"变为"轨迹图 + 时序曲线 + 公式"
- VLM 不再需要猜测"这是静止的还是运动的"——轨迹图直接告诉它
- 公式给了 VLM 可分析的结构，而非纯视觉模式匹配

**对 VLM 能力的现实期望**：
- VLM 的核心价值是**"命名 + 结构化描述"**，而非科研级发现
- VLM 对物理公式的"空间想象力"有限，很可能给出格式化的马后炮（"因为有 sin 所以有振荡"）
- 真正的科研价值来自**导出的人类可读公式集**——VLM 只是给这些公式一个好听的名字
- 如果 VLM 评判质量持续低下，可降级为纯数学评估（去掉 VLM 模块，节省成本）

**成本控制**：
- 每日 VLM 调用上限：100 次
- 超限后仅记录数学特征，不调用 VLM
- 用 `novelty_score` 排序，只发送最 novel 的

---

## 五、显存分配方案

| 模块 | 预算 | 用途 |
|------|------|------|
| 粒子仿真 | 3.5 GB | 100 万粒子状态 + 空间哈希（含桶上限）+ VM 栈 + 中间缓冲 + 营养/废弃物场 |
| 渲染管线 | 0.5 GB | 离屏渲染帧缓冲 |
| 基因组 + 档案缓存 | 0.2 GB | 当前代的 10 个基因组 + 行为向量 |
| Taichi 运行时 | 1 GB | kernel 编译缓存、临时分配 |
| **总计** | **~4.7 GB** | 保留 ~3 GB 余量防 OOM |

**粒子规模弹性**：
- 50 万粒子：显存 ~3.5 GB（安全起步）
- 100 万粒子：显存 ~5 GB（标准配置）
- 200 万粒子：显存 ~8 GB（极限，需关闭 trail 渲染）

---

## 六、运行流程

```
┌─────────────────────────────────────────────────────────────┐
│                     主循环 (main.py)                          │
│                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐    │
│  │ 选择亲本  │───→│ GEP 变异 │───→│ 并行仿真×10          │    │
│  │(MAP-Elites)│   │(公式变异)│    │(Taichi GPU)          │    │
│  └──────────┘    └──────────┘    │                      │    │
│                                  │ 每步:                │    │
│                                  │  1. 空间哈希         │    │
│                                  │  2. 力计算(VM)       │    │
│                                  │  3. 积分             │    │
│                                  │  4. 环境交互 ← 新增  │    │
│                                  │     营养摄取         │    │
│                                  │     代谢消耗         │    │
│                                  │     废弃物沉积       │    │
│                                  │  5. 生命判定         │    │
│                                  └──────────┬───────────┘    │
│                                             │                 │
│                                             ▼                 │
│                                  ┌──────────────────┐         │
│                                  │ 时序特征提取 (×10)│         │
│                                  │ 熵/FFT/角动量/...  │         │
│                                  │ 营养消耗/废弃物/...│         │
│                                  └────────┬─────────┘         │
│                                           │                   │
│                                           ▼                   │
│                                  ┌──────────────────┐         │
│                                  │ 归档到 MAP-Elites │         │
│                                  └────────┬─────────┘         │
│                                           │                   │
│                              ┌────────────┴────────────┐      │
│                              ▼                         ▼      │
│                       ┌────────────┐           ┌────────────┐ │
│                       │ 已知区域    │           │ 空白区域    │ │
│                       │ 静默归档    │           │ 新颖性告警  │ │
│                       │ 记录公式    │           │ 生成轨迹图  │ │
│                       └────────────┘           │ 生成 GIF    │ │
│                                                │ 生成时序曲线 │ │
│                                                └──────┬─────┘ │
│                                                       │       │
│                                                       ▼       │
│                                                ┌────────────┐ │
│                                                │ VLM 评判    │ │
│                                                │ 命名+分析   │ │
│                                                │ 公式解读    │ │
│                                                └──────┬─────┘ │
│                                                       │       │
│                                                       ▼       │
│                                                ┌────────────┐ │
│                                                │ 存入图鉴    │ │
│                                                └────────────┘ │
│                                                               │
│  每 100 代：导出地图快照 + 公式集                              │
│  每 1000 代：保存完整 checkpoint                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 七、数据持久化

### 7.1 SQLite 表结构

**并发策略**：WAL 模式 + 单线程写入。所有仿真进程通过内存队列将结果回传给主进程，主进程单线程执行 DB 写入，避免 `database is locked` 错误。

```sql
-- 3D MAP-Elites 网格（密集竞争）
CREATE TABLE grid_cells (
    grid_key TEXT PRIMARY KEY,       -- "i-j-k"，3 维坐标拼接
    generation INTEGER,
    fitness REAL,
    -- 3D 网格特征（时序不变量）
    f_entropy_mean REAL,
    f_islands_mean REAL,
    f_fft_amp_1 REAL,
    -- 15D 完整特征（JSON 数组，v6: 含环境特征）
    features_15d TEXT,               -- JSON: [entropy_mean, ..., nutrient_consume_mean, waste_peak_mean, energy_skew]
    -- 基因组（人类可读）
    potential_formula TEXT,           -- "1.5 * dist - sin(density * 3.14) - 2.0 * nutrient"（势能公式）
    state_formula TEXT,
    sense_formula TEXT,
    metabolism_formula TEXT,          -- v6: 代谢规则公式
    -- 复现信息
    random_seed INTEGER,             -- 用于重跑复现
    -- 文件路径
    screenshot_path TEXT,
    -- VLM 评判
    vlm_name TEXT,
    vlm_judgment TEXT,
    vlm_score INTEGER,
    created_at TIMESTAMP
);

-- Novelty Archive（开放式归档）
CREATE TABLE novelty_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER,
    fitness REAL,
    novelty_score REAL,              -- k-NN 平均距离
    -- 15D 时序不变量（v6: 含环境特征）
    features_15d TEXT,               -- JSON 数组
    -- 基因组
    potential_formula TEXT,
    state_formula TEXT,
    sense_formula TEXT,
    metabolism_formula TEXT,          -- v6: 代谢规则公式
    -- 复现信息
    random_seed INTEGER,
    -- 文件路径
    screenshot_path TEXT,
    gif_path TEXT,
    -- VLM 评判
    vlm_name TEXT,
    vlm_judgment TEXT,
    vlm_score INTEGER,
    created_at TIMESTAMP
);

-- 演化日志
CREATE TABLE evolution_log (
    generation INTEGER PRIMARY KEY,
    best_fitness REAL,
    avg_fitness REAL,
    novel_count INTEGER,
    dead_count INTEGER,              -- 本代被死寂宇宙过滤的数量
    dormant_count INTEGER,           -- v6: 本代进入休眠状态的粒子数
    vlm_calls INTEGER,
    grid_filled INTEGER,
    archive_size INTEGER,
    avg_nutrient REAL,               -- v6: 平均营养场浓度
    avg_waste REAL,                  -- v6: 平均废弃物场浓度
    avg_energy REAL,                 -- v6: 平均粒子能量
    elapsed_seconds REAL,
    created_at TIMESTAMP
);
```

### 7.2 文件存储

> **核心原则：只存公式和种子，绝对不存高维轨迹。** 需要复现时，用相同随机种子重跑即可。这是参数化生成的最大优势。

```
data/experiments/{experiment_id}/
├── config.yaml
├── archive.db                  # SQLite（grid_cells + novelty_archive + evolution_log）
├── formulas/                   # GEP 公式（人类可读！约 1KB/个）
│   ├── grid/                   # MAP-Elites 网格中的公式
│   │   └── gen_0001_abc123.txt
│   └── archive/                # Novelty Archive 中的公式
│       └── gen_0048_def456.txt
├── screenshots/                # 快照 + 轨迹图（仅新颖性告警时生成）
│   ├── gen_0048_def456.png
│   └── gen_0048_def456_traj.png
├── gifs/                       # 动态 GIF（仅新颖性告警时生成）
│   └── gen_0048_def456.gif
└── checkpoints/
    └── checkpoint_1000.npz
```

**复现机制**：
- 每个基因组存储时附带 `random_seed`（整数）
- 需要复现动态行为时：加载公式 + 种子 → 在 GPU 上重跑 → 得到完整轨迹
- 存储成本：公式 ~1KB + 种子 8 字节 vs 原始轨迹 ~4GB → **节省 400 万倍**

**复现精度声明（重要）**：
- GPU 原子操作的执行顺序不确定 → 浮点加法不满足结合律 → 混沌系统中微小差异在 ~1000 步内被放大
- **可复现**：宏观形态（旋涡、脉冲、聚集模式）——同一公式 + 同一种子，每次重跑都是"同类物种"
- **不可复现**：像素级轨迹——每个粒子的精确位置会有微小偏差
- **对 VLM 的影响**：无。VLM 看的是宏观行为，不需要像素级精确
- **对工程测试的影响**：不要用 assert 测试复现度，用宏观特征（熵、岛屿数等）的相对误差 < 5% 作为通过标准

---

## 八、检查点与恢复

**Checkpoint 内容**：
1. 当前代数
2. 3D MAP-Elites 网格完整状态（含随机种子）
3. Novelty Archive 完整状态（含 15 维行为向量 + 随机种子）
4. 当前代的 10 个活跃基因组（GEP 字节码 + 各自的随机种子）
5. 全局随机种子状态
6. 演化统计日志
7. v6 新增：营养场 + 废弃物场的当前状态（sparse grid 快照）

**恢复流程**：
1. 加载最新 checkpoint
2. 恢复 MAP-Elites 网格
3. 恢复新颖性过滤器
4. 从断点代数继续演化

**频率**：每 1000 代自动保存，收到 SIGINT 时紧急保存。

---

## 九、监控与告警

| 监控项 | 方式 | 阈值/动作 |
|--------|------|-----------|
| GPU 显存 | `nvidia-smi` | > 7.5GB 告警 |
| GPU 温度 | `nvidia-smi` | > 85°C 降速，> 90°C 暂停 |
| 代数进度 | 日志 | 连续 10 代无新颖发现 → 降低新颖性阈值 20% |
| VLM 调用量 | 计数器 | 超日限 → 停止调用 |
| 磁盘空间 | `shutil.disk_usage` | < 5GB → 清理旧 GIF |
| 进程存活 | cron 定期检查 | 崩溃 → 自动重启（从 checkpoint） |
| **营养场浓度** | 每代统计 | avg_nutrient < 0.01 → 增加注入频率或注入量 |
| **废弃物堆积** | 每代统计 | avg_waste > 0.5 → 增加废弃物衰减率 |
| **粒子能量分布** | 每代统计 | avg_energy < 0.1 → 降低基础代谢率 |
| **休眠粒子比例** | 每代统计 | dormant_ratio > 0.3 → 营养注入不足，调整参数 |
| 公式复杂度 | 表达式树深度 | > 20 → 惩罚适应度（防过拟合） |
| NaN 比例 | 每代统计 | > 50% → 跳过本代，降低变异强度 |

**日志格式**：
```
[2026-06-24 15:30:00] gen=12345 | fit=0.847 | novel=2/10 | vlm=1 | grid=342/3375 | archive=1247 | formula="1.5r-sin(d)" | elapsed=2d14h
```

---

## 十、实施阶段

### Phase 0：基础设施（1-2 天）
- [ ] 项目骨架搭建、依赖安装、Taichi CUDA 验证
- [ ] YAML 配置系统
- [ ] SQLite 存储层（WAL 模式 + 单线程写入）
- [ ] 日志系统

### Phase 1：粒子仿真 + VM 压力测试（4-5 天）
- [ ] 粒子状态定义（Taichi fields，含 energy 维度）
- [ ] GPU 原生空间哈希（含桶容量硬上限 128）
- [ ] 栈式 VM 设计：字节码指令集 + GPU 内解释器
- [ ] **VM 压力测试**（最高优先级）：100 万粒子 × 100 步，用 Taichi profiler 检查寄存器溢出
- [ ] 符号求导引擎（CPU 端 AST → dU/dr 的字节码）
- [ ] 粘性阻尼 + 速度/位移限制
- [ ] 单步推进 kernel
- [ ] 简单渲染（密度热力图）
- [ ] 基准性能测试：10 万 / 50 万 / 100 万粒子的 FPS

### Phase 2：GEP 进化引擎（3-4 天）
- [ ] GEP 表达式树定义（安全终端集含 nutrient/waste、安全函数集、常数集）
- [ ] GEP 编码/解码（表达式树 ↔ 字节码）
- [ ] **符号求导**：对 AST 链式法则求偏导，生成 dU/dr 字节码
- [ ] **简化器 + 死代码消除**优化器
- [ ] 字节码安全验证
- [ ] 数值安全：符号求导势能力 + clamp + NaN 检测
- [ ] 遗传算子实现（6 种）
- [ ] 适应度函数（环境驱动 + NaN 惩罚 + parsimony pressure + 死寂宇宙过滤）
- [ ] 主演化循环

### Phase 2.5：环境层（2-3 天）★ v6 新增
- [ ] **营养场**：Taichi sparse grid，扩散+衰减+注入 kernel
- [ ] **废弃物场**：Taichi sparse grid，扩散+衰减 kernel
- [ ] **移动热点**：营养注入点沿螺旋/随机路径漂移
- [ ] **粒子环境交互**：营养摄取、代谢消耗、废弃物沉积、休眠/苏醒/死亡
- [ ] **GEP 终端集扩展**：`nutrient` 和 `waste` 变量接入 VM
- [ ] **环境驱动适应度**：total_energy_harvested + survival_rate + energy_variance
- [ ] 环境层性能测试：营养/废弃物场 kernel 开销 < 10% 总仿真时间

### Phase 3：混合特征档案（2-3 天）
- [ ] **15 维时序不变量**提取（GPU kernel，含 3 维环境特征）
- [ ] 3D MAP-Elites 网格（熵均值×岛屿均值×FFT幅值，15³ = 3375 格子）
- [ ] Novelty Archive（k-NN 新颖性判定，无边界）
- [ ] 混合亲本选择策略（网格 70% + archive 30%）
- [ ] 自适应阈值逻辑
- [ ] 死寂宇宙过滤器（含环境条件：营养消耗率、能量分布偏度）
- [ ] 与演化循环集成

### Phase 4：渲染与 VLM 集成（2 天）
- [ ] 轨迹图渲染（500 帧叠加，颜色编码时间）
- [ ] 时序特征曲线渲染
- [ ] GIF 生成
- [ ] VLM API 调用封装（Ollama + qwen2.5vl:3b）
- [ ] Prompt 模板
- [ ] 成本控制逻辑

### Phase 5：可靠性（1-2 天）
- [ ] Checkpoint 保存/恢复
- [ ] SIGINT 信号处理
- [ ] 监控与告警
- [ ] 自动重启脚本

### Phase 6：可视化与分析（1 天）
- [ ] 3D MAP-Elites 地图可视化
- [ ] Novelty Archive 公式浏览
- [ ] 公式集导出（人类可读的"涌现规则集"）
- [ ] 演化曲线绘制
- [ ] 导出报告功能

**总计：约 14-19 天**（v6 新增 Phase 2.5 环境层 2-3 天）

---

## 十一、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| **Taichi JIT 编译风暴** | ~~已解决~~ | — | 栈式 VM：编译一次，公式变化只改字节码 |
| **GEP 数值爆炸 / NaN 瘟疫** | ~~已解决~~ | — | 四重防护：安全函数集 + 势能力(F=-∇U) + 粘性阻尼 + clamp + NaN 检测 |
| **MAP-Elites 维度灾难** | ~~已解决~~ | — | 3D 网格(3,375 格) + Novelty Archive(无边界) |
| **动量不守恒** | ~~已解决~~ | — | GEP 生成势能 U，力由**符号求导** F=-dU/dr 计算，零精度损失 |
| **空间哈希 O(N²) 坍缩** | ~~已解决~~ | — | 桶容量硬上限 128 粒子/桶，超出拒绝插入 |
| **clamp 高频数值伪影** | ~~已解决~~ | — | 粘性阻尼 γv + 单步最大位移限制(v*dt < 0.5*cell_size) |
| **时序相位不对齐** | ~~已解决~~ | — | 15 维时序不变量（均值/方差/偏度/FFT幅值/自相关/环境特征），对相位不敏感 |
| **死寂宇宙陷阱** | ~~已解决~~ | — | 最小存活测试：存活率<10% 或 熵≈max 或 速度≈0 → fitness=0 |
| **轨迹文件磁盘爆炸** | ~~已解决~~ | — | 只存公式+种子(~1KB)，复现时用种子重跑。比存轨迹节省 400 万倍 |
| **SQLite 并发锁** | ~~已解决~~ | — | 单线程写入(WAL 模式) + 仿真结果通过内存队列回传主进程 |
| **数值求导精度灾难** | ~~已解决~~ | — | 改用**符号求导**：CPU 端对 AST 链式法则求 dU/dr，零精度损失，省一半 VM 开销 |
| **种子复现像素级不可复现** | 已知限制 | 低 | GPU 原子操作顺序不确定 → 浮点不满足结合律 → 混沌放大。宏观形态可复现，像素级不可复现。对 VLM 够用 |
| **VLM 显存不足** | ~~已解决~~ | — | Ollama CPU 模式（`OLLAMA_NUM_GPU=0`），完全不占 GPU 显存 |
| **VM 寄存器溢出** | 中 | 性能下降 10-50x | Phase 1 必须用 Taichi profiler 检查。溢出则改寄存器 VM 或 SIMD 化 |
| Taichi kernel 调试困难 | 中 | 开发效率低 | 先用 CPU 模式调试，确认逻辑后再切 CUDA |
| GEP 搜索空间爆炸 | 高 | 进化停滞 | 限制树深度(<15)，parsimony pressure，安全函数集 |
| GEP 适应度景观不平滑 | 高 | 大部分变异是"死胎" | 固有代价。每代 10 样本 × 长期运行 = 大基数筛选 |
| VM 性能低于预期 | 中 | 每代变慢 | Phase 1 先做 VM 压力测试。太慢则改 SIMD 化 VM 或预编译函数库 |
| GEP 代码膨胀（Bloat） | 中 | 浪费 GPU 算力 | CPU 端死代码消除 + 适应度对有效指令数施加 parsimony pressure |
| 模式坍缩 | 中 | 多样性丧失 | Novelty Archive + MAP-Elites 双保险 |
| GPU OOM | 中 | 进程崩溃 | 保守的显存预算 + 运行时监控 |
| VLM 评判退化为车轱辘话 | 中 | 评判无意义 | 降低期望：VLM 价值是命名+描述；不行就降级为纯数学评估 |
| 长期运行内存泄漏 | 中 | 进程崩溃 | 定期 checkpoint + 自动重启 |
| **环境层 OOM** | 中 | 显存不足 | 营养/废弃物场用 sparse grid，只分配被粒子访问的格子 |
| **营养场数值溢出** | 低 | NaN 瘟疫 | 每步 clamp 营养/废弃物场到 [0, max_value] |
| **环境层性能开销** | 中 | 每代变慢 | 扩散/衰减 kernel 与粒子 kernel 合并，减少显存带宽瓶颈 |
| **休眠孢子堆积** | 低 | 大量僵尸粒子 | 最大休眠时间 600 tick，超时强制清理 |

---

## 十二、配置示例

```yaml
# config/default.yaml
experiment:
  name: "hard-mode-v1"
  seed: 42

simulation:
  num_particles: 500000
  particle_state_dim: 5        # x, y, vx, vy, energy（v6: 新增能量维度）
  dt: 0.01
  steps_per_eval: 50000
  boundary: "periodic"
  damping_gamma: 0.1         # 粘性阻尼系数
  bucket_max: 128            # 空间哈希桶容量硬上限

environment:                    # v6: 环境层
  nutrient_diffuse_rate: 0.08   # 营养扩散速率
  nutrient_decay_rate: 0.001    # 营养衰减率（每步）
  nutrient_inject_interval: 60  # 营养注入间隔（步）
  nutrient_patch_count: 5       # 每次注入斑块数
  nutrient_patch_amount: 1.5    # 斑块中心浓度
  nutrient_drift_speed: 0.002   # 热点漂移速度（弧度/步）
  waste_production_rate: 0.15   # 每单位营养消耗产生的废弃物量
  waste_diffuse_rate: 0.05      # 废弃物扩散速率
  waste_decay_rate: 0.005       # 废弃物衰减率（每步）
  waste_metabolism_factor: 2.0  # 毒素区代谢放大系数
  waste_repel_strength: 0.8     # 废弃物排斥强度
  base_metabolism: 0.01         # 基础代谢消耗（每步）
  move_cost: 0.005              # 移动代谢成本（每单位速度）
  absorb_rate: 0.5              # 营养吸收效率
  dormant_metabolism: 0.001     # 休眠状态代谢率
  max_dormant_ticks: 600        # 最大休眠时间（超时则死亡）

gep:
  terminal_set: ["dist", "density", "speed", "angle", "state_0", "state_1", "state_2", "state_3", "neighbor_count", "nutrient", "waste"]
  function_set: ["+", "-", "*", "sin", "cos", "tanh", "sqrt", "abs", "max", "min"]  # 安全子集，无 exp/pow/div/log
  constant_range: [-5.0, 5.0]
  max_tree_depth: 12
  head_length: 8
  bytecode_length: 128       # 字节码最大指令数
  vm_stack_depth: 16         # 虚拟机栈深度

evolution:
  population_size: 10
  mutation_rates:
    point_mutation: 0.30
    constant_finetune: 0.20
    is_transposition: 0.15
    ris_transposition: 0.10
    one_point_recombination: 0.15
    two_point_recombination: 0.10
  parsimony_pressure: 0.001

map_elites:
  # 3D 精英网格（密集竞争）
  grid_features: ["entropy", "islands", "fft_freq"]
  resolution_per_dim: 15     # 15³ = 3,375 格子
  # 15D 特征用于记录（不用于网格索引）
  all_features: ["entropy", "islands", "fft_freq", "angular_skew", "density_lap_var", "survival_decay",
                  "nutrient_consume_mean", "waste_peak_mean", "energy_skew"]  # v6: 环境特征

novelty:
  # Novelty Archive（无边界开放式归档）
  behavior_vector_dim: 15    # 15 维时序不变量（v6: 含 3 维环境特征）
  k_neighbors: 15            # k-NN 近邻数
  threshold_adaptive: true
  stale_generations: 10      # 连续 N 代无新颖 → 降阈值 20%
  sample_interval: 500       # 每 500 步采样一次特征
  # 亲本选择混合比例
  grid_selection_prob: 0.7   # 70% 从网格选，30% 从 archive 选
  # 死寂宇宙过滤
  min_survival_rate: 0.1     # 存活率 < 10% → fitness = 0
  min_speed_variance: 0.001  # 速度方差 < 0.001 → fitness = 0
  max_entropy_ratio: 0.95    # 熵 > 95% 最大熵 → fitness = 0

rendering:
  resolution: [256, 256]
  trajectory_frames: 500     # 轨迹图叠加帧数
  gif_frames: 200            # GIF 帧数
  gif_fps: 10

safety:
  clamp_min: -100.0          # 势能 clamp 下界
  clamp_max: 100.0           # 势能 clamp 上界
  nan_penalty: 0.0           # NaN 个体适应度直接赋此值
  max_force: 10.0            # 单粒子最大力
  max_speed: 5.0             # 单粒子最大速度
  max_displacement_ratio: 0.5 # 单步最大位移 = ratio * cell_size

vlm:
  provider: "ollama"             # ollama | local | openai | anthropic
  model: "qwen2.5vl:3b"         # Ollama 本地视觉模型（~3.2GB）
  fallback_model: "llava:7b"    # 兜底模型
  base_url: "http://localhost:11434"  # Ollama 端点
  daily_limit: 100
  # 如果用 API：
  # provider: "openai"
  # model: "gpt-4o"
  # api_key: "sk-..."

checkpoint:
  interval: 1000
  max_keep: 5

monitoring:
  gpu_temp_threshold: 85
  disk_min_gb: 5
  stale_generations: 10
  formula_max_depth: 20
```

---

## 附录 A：与旧方案的关键差异

| 维度 | v1 | v2 | v3 | v4 | v5 | v6（本文档） |
|------|----|----|----|----|----|----|
| 计算框架 | JAX | Taichi | Taichi | Taichi | Taichi | Taichi |
| 公式执行 | — | 动态编译 | 栈式 VM | 栈式 VM | 栈式 VM | 栈式 VM |
| 力计算 | 直接输出 | 直接输出 | 直接输出 | 数值求导 F=-∇U | 符号求导 F=-dU/dr | 符号求导 F=-dU/dr |
| 求导精度 | — | — | — | float32 精度灾难 | 零精度损失 | 零精度损失 |
| 空间哈希 | JAX padding | Taichi | Taichi | Taichi+桶上限 | Taichi+桶上限 | Taichi+桶上限 |
| 基因组 | MLP 权重 | GEP 力 | GEP 力 | GEP 势能 U | GEP 势能 U | GEP 势能 U（含环境变量） |
| 数值安全 | 无 | 无 | 三重 | 五重 | 五重+符号求导 | 五重+符号求导 |
| 特征 | 2D | 6D | 3D+archive | 12D 不变量 | 12D 不变量 | **15D 不变量（含环境特征）** |
| 相位敏感 | — | — | — | 不敏感 | 不敏感 | 不敏感 |
| 死寂过滤 | 无 | 无 | 无 | 有 | 有 | 有（含环境条件） |
| 存储 | 截图 | 轨迹 | 轨迹 | 公式+种子 | 公式+种子 | 公式+种子 |
| 复现精度 | — | — | — | 像素级(伪) | 宏观可复现 | 宏观可复现 |
| VLM | API | API | API | API | 本地 Qwen2-VL-2B | **本地 Qwen2.5-VL-3B** |
| VLM 显存 | — | — | — | — | 时分复用 | Ollama CPU 模式 |
| **环境层** | 无 | 无 | 无 | 无 | 无 | **营养场+废弃物场** |
| **适应度** | 人工定义 | 人工定义 | 人工定义 | 人工定义 | 人工定义 | **环境自然选择+人工辅助** |
| **涌现驱动力** | 力交互 | 力交互 | 力交互 | 力交互 | 力交互 | **力交互+环境反馈** |
| 哲学 | 黑盒 | 可解释 | 可解释 | 可解释 | 可解释 | 可解释+环境驱动 |

## 附录 B：从当前代码库迁移

当前 `main` 分支的浏览器方案（p5.js + Puppeteer）与 Hard Mode 无代码复用关系，是完全独立的实现。建议：

1. 在 `hard` 分支上从零开始
2. `main` 分支保留作为参考（其中的涌现理论和参数空间探索经验仍有价值）
3. 不删除旧代码，但新代码不依赖它
