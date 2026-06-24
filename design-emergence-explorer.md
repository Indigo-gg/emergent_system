# 涌现参数探索器（Emergence Explorer）设计方案

> 用多模态 LLM 作为"涌现观察者"，自动搜索参数空间，发现并记录涌现现象。
> 全过程可溯源：每一轮的参数、截图、LLM 反馈完整保存。

---

## 一、系统目标

手动调参找涌现是盲人摸象。本系统将这个过程自动化：

1. **自动运行**：给定一组参数，自动初始化引擎、运行仿真、截图
2. **自动评估**：将截图发给多模态 LLM，由其评判"是否构成涌现"
3. **自动记录**：参数、截图、LLM 反馈、评分完整存档，可随时回溯
4. **智能搜索**：LLM 根据历史实验记录，建议下一轮参数调整方向

核心理念：**LLM 既是观察者，也是实验科学家**。

### 已知风险与防线

在设计之初，我们识别了 4 个致命风险，并在架构中内置了对应防线：

| 风险 | 本质 | 防线 |
|---|---|---|
| **时间坍缩** | 单张截图无法证明结构稳定 | 多帧时序切片 + 时序方差统计 |
| **空想性错视** | LLM 在噪声中"看到"不存在的模式 | 同步序参量 R 等硬数学指标锚定 |
| **相变悬崖** | 非线性参数空间中 LLM 线性推理失效 | 遗传算法精细搜索 + 相变边界探测提示 |
| **模式匹配退化** | 预设 rubric 错过未知涌现 | 新颖性（Novelty）维度 + 异常标记 |

以下设计中所有标注 `[RISK-1]` `[RISK-2]` `[RISK-3]` `[RISK-4]` 的部分，都是针对这些风险的防线实现。

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Emergence Explorer                        │
│                                                              │
│  ┌────────────┐     ┌──────────────┐     ┌──────────────┐  │
│  │  Parameter  │────▶│  Simulation  │────▶│  Screenshot  │  │
│  │  Scheduler  │     │  Runner      │     │  Capture     │  │
│  │             │     │  (browser)   │     │  (canvas)    │  │
│  └─────┬──────┘     └──────────────┘     └──────┬───────┘  │
│        │                                         │          │
│        │                                         ▼          │
│        │                                  ┌──────────────┐  │
│        │                                  │   LLM API    │  │
│        │◀─────────────────────────────────│   Evaluator  │  │
│        │         score + next_params      │              │  │
│        │                                  └──────────────┘  │
│        ▼                                                     │
│  ┌──────────────────────────────────────┐                   │
│  │         Experiment Database          │                   │
│  │  runs/manifest.jsonl                 │                   │
│  │  runs/<id>/params.json               │                   │
│  │  runs/<id>/screenshot.png            │                   │
│  │  runs/<id>/llm-response.json         │                   │
│  │  runs/<id>/config-snapshot.json      │                   │
│  └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### 组件职责

| 组件 | 职责 | 运行环境 |
|---|---|---|
| **Parameter Scheduler** | 决定下一轮参数组合，维护历史记录 | Node.js 脚本 |
| **Simulation Runner** | 用给定参数初始化引擎并运行 N 帧 | 浏览器（Puppeteer 控制） |
| **Screenshot Capture** | 从 p5.js canvas 截取累积渲染图 | 浏览器内 `canvas.toDataURL()` |
| **LLM Evaluator** | 将截图+rubric 发给 LLM，解析评分 | Node.js 脚本，调用 LLM API |
| **Experiment Database** | 存储所有实验数据，保证可溯源 | 本地文件系统 |

---

## 三、参数空间定义

### 3.1 核心参数（Phase 1：振子同步）

这些参数直接控制涌现行为，是探索的主要目标：

```javascript
// src/explorer/param-space.js

export const PARAM_SPACE = {
  // --- 涌现核心参数 ---
  temperature: {
    min: 0.0,
    max: 2.0,
    step: 0.05,
    description: '热扰动强度。0 = 完全确定性，2 = 高度随机',
    category: 'emergence',
  },
  couplingStrength: {
    min: 0.0,
    max: 5.0,
    step: 0.1,
    description: 'Kuramoto 同步耦合强度。0 = 无同步倾向，5 = 强同步',
    category: 'emergence',
  },
  perceptionRadius: {
    min: 10,
    max: 200,
    step: 5,
    description: '粒子感知半径（像素）。影响邻居数量和集群大小',
    category: 'emergence',
  },
  decayRate: {
    min: 0.90,
    max: 0.999,
    step: 0.005,
    description: '信息素/痕迹衰减率。越接近1，痕迹越持久',
    category: 'emergence',
  },

  // --- 粒子参数 ---
  particleCount: {
    min: 100,
    max: 5000,
    step: 100,
    description: '粒子总数。影响密度和计算量',
    category: 'particles',
  },
  initialSpeed: {
    min: 0.1,
    max: 3.0,
    step: 0.1,
    description: '粒子初始速度范围',
    category: 'particles',
  },

  // --- 物理参数 ---
  friction: {
    min: 0.90,
    max: 0.999,
    step: 0.005,
    description: '速度阻尼。越接近1，惯性越大',
    category: 'physics',
  },
  maxSpeed: {
    min: 0.5,
    max: 5.0,
    step: 0.25,
    description: '最大速度钳制',
    category: 'physics',
  },

  // --- 渲染参数（不影响仿真，只影响截图外观） ---
  trailAlpha: {
    min: 1,
    max: 80,
    step: 5,
    description: '背景透明度。越低，拖尾越长',
    category: 'render',
  },
  particleSize: {
    min: 1,
    max: 8,
    step: 1,
    description: '粒子绘制尺寸',
    category: 'render',
  },
};
```

### 3.2 参数配置文件格式

每一轮实验的参数用一个 JSON 文件描述：

```json
// runs/0001/params.json
{
  "runId": "0001",
  "timestamp": "2026-06-23T14:30:00.000Z",
  "round": 1,
  "strategy": "random",
  "params": {
    "temperature": 0.35,
    "couplingStrength": 1.2,
    "perceptionRadius": 60,
    "decayRate": 0.97,
    "particleCount": 1000,
    "initialSpeed": 0.5,
    "friction": 0.98,
    "maxSpeed": 2.0,
    "trailAlpha": 20,
    "particleSize": 3
  },
  "derived": {
    "cellSize": 60,
    "canvasW": 900,
    "canvasH": 900,
    "seed": 42,
    "simFrames": 1000
  }
}
```

`derived` 字段记录由核心参数推导出的配置，确保完全可复现。

---

## 四、仿真运行器（Simulation Runner）

### 4.1 运行环境

仿真运行在浏览器中（p5.js 依赖 DOM），由 Puppeteer 脚本控制：

```
Node.js (Puppeteer)
    │
    ├── 启动 headless Chrome
    ├── 打开 explorer.html
    ├── page.evaluate(runSimulation, params)
    │       ├── 初始化 ParticleState
    │       ├── 初始化 SpatialHash
    │       ├── 注册 KuramotoPlugin + NoisePlugin
    │       ├── 循环 engine.update() × N 帧
    │       └── 返回 canvas.toDataURL() (base64 截图)
    └── 保存截图到 runs/<id>/screenshot.png
```

### 4.2 explorer.html — 专用仿真页面

这是一个独立的 HTML 页面，专门为自动化探索设计。它暴露一个全局函数供 Puppeteer 调用：

```javascript
// src/explorer/runner.js
// 在 explorer.html 中加载，暴露给 Puppeteer

import { ParticleState } from '../state.js';
import { SpatialHash } from '../spatial-hash.js';
import { Engine } from '../engine.js';
import { PluginRegistry } from '../plugin.js';
import { createRNG } from '../prng.js';
import { KuramotoPlugin } from '../plugins/kuramoto.js';
import { NoisePlugin } from '../plugins/noise.js';

/**
 * 由 Puppeteer 通过 page.evaluate() 调用。
 *
 * [RISK-1 修复] 返回多帧时序切片 + 时序方差统计，而非单张截图。
 * [RISK-2 修复] 返回同步序参量 R 等硬数学指标。
 *
 * @param {Object} params - 完整的参数配置
 * @returns {Object} { screenshots: base64[], stats: {...} }
 */
window.runEmergenceExperiment = async function (params) {
  const {
    temperature, couplingStrength, perceptionRadius, decayRate,
    particleCount, initialSpeed, friction, maxSpeed,
    trailAlpha, particleSize,
    canvasW = 900, canvasH = 900, seed = 42,
    simFrames = 1000,
  } = params;

  const rng = createRNG(seed);

  // --- 初始化引擎 ---
  const state = new ParticleState(particleCount);
  for (let i = 0; i < particleCount; i++) {
    const x = rng() * canvasW;
    const y = rng() * canvasH;
    const angle = rng() * Math.PI * 2;
    const speed = initialSpeed * (0.5 + rng() * 0.5);
    state.spawn(x, y, Math.cos(angle) * speed, Math.sin(angle) * speed);
    state.phase[i] = rng() * Math.PI * 2;
  }

  const cellSize = perceptionRadius;
  const spatialHash = new SpatialHash(cellSize, canvasW, canvasH);

  const pluginRegistry = new PluginRegistry();
  pluginRegistry.register(new KuramotoPlugin(couplingStrength, perceptionRadius));
  pluginRegistry.register(new NoisePlugin(temperature));

  const config = {
    dt: 1 / 60, maxSpeed, maxForce: 0.1, friction,
    canvasW, canvasH, perceptionRadius, decayRate,
  };
  const engine = new Engine(config, state, spatialHash, pluginRegistry);

  // --- 创建离屏 canvas ---
  const canvas = document.createElement('canvas');
  canvas.width = canvasW;
  canvas.height = canvasH;
  const ctx = canvas.getContext('2d');

  // ============================================================
  // [RISK-1 修复] 时序切片采集点
  // 前 70% 是预热期（burn-in），只在最后 30% 的稳定期考察结构
  // ============================================================
  const slicePoints = [
    Math.floor(simFrames * 0.70),
    Math.floor(simFrames * 0.80),
    Math.floor(simFrames * 0.90),
    simFrames,
  ];
  const sliceScreenshots = [];
  let nextSliceIdx = 0;

  // ============================================================
  // [RISK-2 修复] 时序统计采集窗口
  // 最后 200 帧每 10 帧采样一次，用于计算方差
  // ============================================================
  const statsWindowStart = simFrames - 200;
  const statsSampleInterval = 10;
  const timeSeries = {
    sync_R: [],        // 同步序参量
    avgSpeed: [],      // 平均速度
    clusterCount: [],  // 簇数量（粗估）
  };

  // --- 运行仿真 ---
  for (let frame = 0; frame < simFrames; frame++) {
    engine.update(1 / 60);

    // 采集时序切片截图
    if (nextSliceIdx < slicePoints.length && frame === slicePoints[nextSliceIdx]) {
      sliceScreenshots.push(renderFrame(ctx, state, canvasW, canvasH, maxSpeed, particleSize));
      nextSliceIdx++;
    }

    // 采集时序统计
    if (frame >= statsWindowStart && frame % statsSampleInterval === 0) {
      timeSeries.sync_R.push(computeSyncR(state, spatialHash, perceptionRadius));
      timeSeries.avgSpeed.push(computeAvgSpeed(state));
      timeSeries.clusterCount.push(estimateClusterCount(state, spatialHash, perceptionRadius));
    }
  }

  // --- 最终帧的 trail 累积渲染 ---
  ctx.fillStyle = 'rgba(0, 0, 0, 1)';
  ctx.fillRect(0, 0, canvasW, canvasH);
  const trailFrames = 60;
  for (let frame = 0; frame < trailFrames; frame++) {
    engine.update(1 / 60);
    ctx.fillStyle = `rgba(0, 0, 0, ${trailAlpha / 255})`;
    ctx.fillRect(0, 0, canvasW, canvasH);
    for (let i = 0; i < state.count; i++) {
      const speed = Math.sqrt(state.vx[i] ** 2 + state.vy[i] ** 2);
      const hue = map(speed, 0, maxSpeed, 220, 30);
      const sat = map(speed, 0, maxSpeed, 60, 100);
      const bri = map(speed, 0, maxSpeed, 40, 100);
      ctx.fillStyle = `hsl(${hue}, ${sat}%, ${bri}%)`;
      ctx.beginPath();
      ctx.arc(state.x[i], state.y[i], particleSize, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  const trailScreenshot = canvas.toDataURL('image/png');

  // ============================================================
  // [优化] 拼图（Sprite Sheet）
  // 将 4 张时序切片 + 1 张 trail 拼成一张全景大图
  // 左侧 2x2 网格放时序切片（带时间戳标注），右侧放 trail 大图
  // 这样 LLM 只需处理 1 张图，成本降低 80%，且时序对比更直观
  // ============================================================
  const thumbW = Math.floor(canvasW / 2);
  const thumbH = Math.floor(canvasH / 2);
  const compositeW = canvasW + canvasW; // 左侧 4 缩略图 + 右侧 1 大图
  const compositeH = canvasH;
  const compositeCanvas = document.createElement('canvas');
  compositeCanvas.width = compositeW;
  compositeCanvas.height = compositeH;
  const compCtx = compositeCanvas.getContext('2d');

  compCtx.fillStyle = 'rgb(0, 0, 0)';
  compCtx.fillRect(0, 0, compositeW, compositeH);

  // 左侧：4 张时序切片（2x2 网格）
  const sliceLabels = ['70%', '80%', '90%', '100%'];
  for (let i = 0; i < sliceScreenshots.length; i++) {
    const img = new Image();
    img.src = sliceScreenshots[i];
    await new Promise(resolve => { img.onload = resolve; });

    const dx = (i % 2) * thumbW;
    const dy = Math.floor(i / 2) * thumbH;
    compCtx.drawImage(img, dx, dy, thumbW, thumbH);

    // 时间戳标注
    compCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    compCtx.fillRect(dx + 5, dy + 5, 100, 28);
    compCtx.fillStyle = 'white';
    compCtx.font = 'bold 16px monospace';
    compCtx.fillText(`Time: ${sliceLabels[i]}`, dx + 12, dy + 24);
  }

  // 右侧：trail 大图
  const trailImg = new Image();
  trailImg.src = trailScreenshot;
  await new Promise(resolve => { trailImg.onload = resolve; });
  compCtx.drawImage(trailImg, canvasW, 0, canvasW, canvasH);

  // 右侧标注
  compCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
  compCtx.fillRect(canvasW + 5, 5, 160, 28);
  compCtx.fillStyle = 'white';
  compCtx.font = 'bold 16px monospace';
  compCtx.fillText('Trail (60 frames)', canvasW + 12, 24);

  const compositeScreenshot = compositeCanvas.toDataURL('image/png');

  // --- 最终统计（包含硬数学指标） ---
  const finalSyncR = computeSyncR(state, spatialHash, perceptionRadius);
  const stats = {
    // 基础统计
    avgSpeed: computeAvgSpeed(state),
    tickCount: engine.tickCount,

    // [RISK-2] 硬数学指标 — LLM 不能无视这些
    sync_order_parameter: finalSyncR,
    cluster_count: estimateClusterCount(state, spatialHash, perceptionRadius),

    // [RISK-1] 时序方差 — 如果方差大，说明结构不稳定
    temporal_variance: {
      sync_R_variance: variance(timeSeries.sync_R),
      avgSpeed_variance: variance(timeSeries.avgSpeed),
      clusterCount_variance: variance(timeSeries.clusterCount),
      sync_R_trend: trend(timeSeries.sync_R),  // 'rising' | 'falling' | 'stable'
    },
  };

  return {
    // 原始切片（存档用）
    screenshots: [...sliceScreenshots, trailScreenshot],
    // 拼图（发给 LLM 评估用，1 张图替代 5 张）
    composite_screenshot: compositeScreenshot,
    stats,
  };
};

// --- 渲染一帧的瞬时状态（无 trail） ---
function renderFrame(ctx, state, w, h, maxSpeed, particleSize) {
  ctx.fillStyle = 'rgb(0, 0, 0)';
  ctx.fillRect(0, 0, w, h);
  for (let i = 0; i < state.count; i++) {
    const speed = Math.sqrt(state.vx[i] ** 2 + state.vy[i] ** 2);
    const hue = map(speed, 0, maxSpeed, 220, 30);
    const sat = map(speed, 0, maxSpeed, 60, 100);
    const bri = map(speed, 0, maxSpeed, 40, 100);
    ctx.fillStyle = `hsl(${hue}, ${sat}%, ${bri}%)`;
    ctx.beginPath();
    ctx.arc(state.x[i], state.y[i], particleSize, 0, Math.PI * 2);
    ctx.fill();
  }
  return ctx.canvas.toDataURL('image/png');
}

// [RISK-2] 局部平均同步序参量 (Local Order Parameter)
//
// 全局 R 的陷阱：如果系统涌现出 3 个各自同步但互不同步的簇
// （相位分别为 0°, 120°, 240°），全局 R = 0，会误判为"无同步"。
// 局部 R 只看每个粒子与感知半径内邻居的同步度，再求全图平均。
// 这是学术界衡量 Spatial Kuramoto 模型的标准做法。
//
// R_local ≈ 0 → 邻居间无同步, R_local ≈ 1 → 局部完全同步
function computeSyncR(state, spatialHash, radius) {
  const n = state.count;
  if (n === 0) return 0;

  let totalLocalR = 0;
  let validParticles = 0;

  for (let i = 0; i < n; i++) {
    let sumCos = Math.cos(state.phase[i]);  // 包含自身
    let sumSin = Math.sin(state.phase[i]);
    let neighborCount = 1;

    spatialHash.queryState(state.x[i], state.y[i], radius, state, (j, _d2) => {
      sumCos += Math.cos(state.phase[j]);
      sumSin += Math.sin(state.phase[j]);
      neighborCount++;
    });

    if (neighborCount <= 1) continue; // 孤立粒子不计入

    const localR = Math.sqrt(sumCos * sumCos + sumSin * sumSin) / neighborCount;
    totalLocalR += localR;
    validParticles++;
  }

  return validParticles > 0 ? (totalLocalR / validParticles) : 0;
}

function computeAvgSpeed(state) {
  let total = 0;
  for (let i = 0; i < state.count; i++) {
    total += Math.sqrt(state.vx[i] ** 2 + state.vy[i] ** 2);
  }
  return total / state.count;
}

// 粗估簇数量：统计空间 hash 中被占据的格子数 / 平均每格粒子数
function estimateClusterCount(state, spatialHash, radius) {
  let occupiedCells = 0;
  for (let c = 0; c < spatialHash.numCells; c++) {
    if (spatialHash.cellStart[c] !== -1) occupiedCells++;
  }
  // 粗糙估计：簇数 ≈ 占据格子数 / (感知半径覆盖的格子数)
  const cellsPerCluster = Math.max(1, Math.ceil(radius / spatialHash.cellSize));
  return Math.max(1, Math.round(occupiedCells / (cellsPerCluster * cellsPerCluster)));
}

function variance(arr) {
  if (arr.length < 2) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return arr.reduce((sum, v) => sum + (v - mean) ** 2, 0) / arr.length;
}

// 简单线性趋势：比较前半和后半的均值
function trend(arr) {
  if (arr.length < 4) return 'stable';
  const half = Math.floor(arr.length / 2);
  const firstHalf = arr.slice(0, half).reduce((a, b) => a + b, 0) / half;
  const secondHalf = arr.slice(half).reduce((a, b) => a + b, 0) / (arr.length - half);
  const diff = (secondHalf - firstHalf) / (firstHalf || 1);
  if (diff > 0.1) return 'rising';
  if (diff < -0.1) return 'falling';
  return 'stable';
}

function map(value, start1, stop1, start2, stop2) {
  return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
}
```

### 4.3 Puppeteer 自动化脚本

```javascript
// scripts/run-experiment.js

import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';

const RUNS_DIR = path.resolve('runs');

export async function runExperiment(params, roundId) {
  const runId = String(roundId).padStart(4, '0');
  const runDir = path.join(RUNS_DIR, runId);
  fs.mkdirSync(runDir, { recursive: true });

  // 1. 保存参数（可溯源）
  const paramsFile = {
    runId,
    timestamp: new Date().toISOString(),
    round: roundId,
    params: params,
    derived: {
      cellSize: params.perceptionRadius,
      canvasW: 900,
      canvasH: 900,
      seed: 42,
      simFrames: 1000,
    },
  };
  fs.writeFileSync(
    path.join(runDir, 'params.json'),
    JSON.stringify(paramsFile, null, 2)
  );

  // 2. 启动浏览器，运行仿真
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto('http://localhost:5173/explorer.html');

  const result = await page.evaluate((p) => {
    return window.runEmergenceExperiment(p);
  }, { ...params, ...paramsFile.derived });

  // 3. [RISK-1] 保存原始切片截图（存档用）
  const sliceLabels = ['t70', 't80', 't90', 't100', 'trail'];
  for (let i = 0; i < result.screenshots.length; i++) {
    const label = sliceLabels[i] || `slice${i}`;
    const base64Data = result.screenshots[i].replace(/^data:image\/png;base64,/, '');
    fs.writeFileSync(path.join(runDir, `screenshot_${label}.png`), base64Data, 'base64');
  }

  // 4. [优化] 保存拼图（Sprite Sheet）—— 发给 LLM 评估用
  const compositeData = result.composite_screenshot.replace(/^data:image\/png;base64,/, '');
  fs.writeFileSync(path.join(runDir, 'screenshot_composite.png'), compositeData, 'base64');

  // 5. 保存运行统计（包含硬数学指标和时序方差）
  fs.writeFileSync(
    path.join(runDir, 'stats.json'),
    JSON.stringify(result.stats, null, 2)
  );

  await browser.close();

  // 返回 trail 截图（主评估用）+ 所有切片（时序验证用）+ 统计
  return {
    runId,
    runDir,
    composite_screenshot: result.composite_screenshot,  // 拼图（发给 LLM）
    screenshots: result.screenshots,                     // 原始切片（存档）
    stats: result.stats,
  };
}
```

---

## 五、LLM 评估器

### 5.1 评估 Prompt 设计

这是整个系统的灵魂。Prompt 需要精心设计，让 LLM 既能识别涌现，又能给出可量化的分数。

```javascript
// src/explorer/llm-evaluator.js

/**
 * [RISK-1 修复] 接收多帧截图而非单张。
 * [RISK-2 修复] 在 prompt 中嵌入硬数学指标约束。
 * [RISK-4 修复] 增加新颖性（Novelty）评分维度。
 */
export function buildEvalPrompt(params, stats, previousRuns = []) {
  const historySection = buildHistorySummary(previousRuns);

  // [RISK-2] 硬数学约束：根据 sync_order_parameter 强制限制评分
  const hardConstraints = buildHardConstraints(stats);

  // [RISK-4] 新颖性参考：最近 5 轮的描述，用于判断是否"打破既有分类"
  const noveltyReference = buildNoveltyReference(previousRuns);

  return `你是一个复杂系统研究专家，正在评估一个人工粒子宇宙的涌现现象。

## 实验背景
这是一个基于 Kuramoto 振子同步 + 热扰动的粒子系统。每个粒子有：
- 位置和速度
- 内部振荡相位（phase）
- 感知范围内邻居的同步倾向

系统参数：
${formatParams(params)}

运行统计（包含硬数学指标）：
${formatStats(stats)}

${historySection}

${hardConstraints}

## 你的任务

你收到了一张**拼图（Sprite Sheet）**，布局如下：
- **左侧**（2×2 网格）：仿真运行到 70%、80%、90%、100% 时刻的**瞬时快照**（时序切片），每个缩略图带有时间戳标注
- **右侧**：最终 60 帧的**累积轨迹图**（trail 渲染），尺寸为左侧单张的 4 倍

**[RISK-1] 时序评估要求**：请对比左侧 4 张时序切片判断结构是否在稳定期保持一致。
- 如果四张切片中的宏观结构形态一致或呈周期性变化 → 结构稳定
- 如果四张切片差异巨大 → 结构不稳定，Structural Stability 最高 3 分

### 评分维度（每项 1-10 分）

**1. 空间聚集度 (Spatial Clustering)**
- 1分：粒子完全均匀随机分布，无任何聚集倾向
- 3分：有零星的小团，但不稳定
- 5分：清晰可见的多个粒子簇，边界可辨
- 7分：大尺度的聚集结构，簇间有明确的间距
- 10分：高度自组织的多层级聚落，像星系团一样层次分明

**2. 结构稳定性 (Structural Stability)** ← 必须参考时序切片
- 1分：四张时序切片完全不同，结构瞬息万变
- 3分：有短暂的结构闪现但在某张切片中已消散
- 5分：主要结构在至少 3 张切片中持续存在
- 7分：结构在所有 4 张切片中形态一致
- 10分：结构如同凝固的雕塑，在整个时间窗口内完全不变

**3. 同步程度 (Synchronization)**
- 1分：所有粒子各自随机运动，无协调
- 3分：偶尔有小群粒子同向移动
- 5分：可以观察到局部区域的同步流动
- 7分：大面积的同步振荡或同向流动，像鸟群一样
- 10分：整个系统展现出壮观的集体同步行为

**4. 层级复杂度 (Hierarchical Complexity)**
- 1分：单一尺度，没有内部结构
- 3分：可以看到簇的边界，但簇内无结构
- 5分：簇内有子结构，或簇之间有连接通道
- 7分：多层级自相似结构，像分形一样
- 10分：令人惊叹的多尺度复杂性，每个放大倍率都有新结构

**5. 美学涌现 (Aesthetic Emergence)**
- 1分：纯噪声，毫无美感
- 3分：有一点模式但很粗糙
- 5分：有吸引力的图案，让人多看几眼
- 7分：令人印象深刻，有"生命感"
- 10分：让人屏息的有机结构，像是来自另一个宇宙的生命

**6. [RISK-4] 新颖性 (Novelty)** ← 对比历史描述判断
- 1-3分：与历史实验中的模式类似，无新意
- 4-6分：在已知模式基础上有变化，但本质相同
- 7-9分：展现出前所未见的拓扑结构或动力学形态
- 10分：完全打破既有分类，令人震撼的全新涌现形态

### 负面清单（以下情况不是涌现，请扣分）
- 粒子全部堆积在画布边缘或角落 → 最高 2 分
- 所有粒子完全静止不动 → 最高 1 分
- 粒子呈现完美的几何对称（规则圆环、正方形网格）→ 这是死板的晶体态，最高 3 分
- 数值溢出导致的异常图案（粒子飞出画布、NaN 传播）→ 最高 1 分

### [RISK-3] 相变边界提示
你正在评估一个非线性系统。涌现往往发生在"有序"与"混沌"的刃尖上。
- 如果数学指标显示系统处于临界状态（R 在 0.3-0.7 之间），这可能是最有趣的区域
- 如果你发现两组相近参数的得分差异巨大，说明你找到了相变边界

${noveltyReference}

## 输出格式
请严格按照以下 JSON 格式输出，不要输出其他内容：

\`\`\`json
{
  "scores": {
    "spatial_clustering": <1-10>,
    "structural_stability": <1-10>,
    "synchronization": <1-10>,
    "hierarchical_complexity": <1-10>,
    "aesthetic_emergence": <1-10>,
    "novelty": <1-10>
  },
  "composite_score": <加权平均，权重见下方>,
  "emergence_level": "<none|weak|moderate|strong|extraordinary>",
  "description": "<一句话描述你看到的现象，不超过50字>",
  "observations": ["<观察1>", "<观察2>", ...],
  "temporal_assessment": "<基于时序切片的结构稳定性判断，30字以内>",
  "is_false_positive": <true|false>,
  "false_positive_reason": "<如果是伪涌现，解释原因；否则为 null>",
  "is_novel": <true|false>,
  "novelty_note": "<如果 is_novel=true，描述新颖之处；否则为 null>",
  "suggestion": {
    "direction": "<建议下一轮参数调整方向>",
    "param_hints": {
      "<参数名>": "<增大|减小|保持>"
    }
  }
}
\`\`\`

加权公式：composite_score = spatial×0.15 + stability×0.2 + sync×0.2 + hierarchy×0.1 + aesthetic×0.15 + novelty×0.2

请基于截图和数据中的实际证据打分，不要猜测。如果你不确定某个维度，偏向保守评分。`;
}

/**
 * [RISK-2] 根据硬数学指标生成评分约束。
 * 用物理量锚定 LLM 的评分，防止空想性错视。
 */
function buildHardConstraints(stats) {
  const constraints = [];

  // 同步序参量约束（使用局部平均 R，避免多簇互不同步时的误判）
  const R = stats.sync_order_parameter;
  if (R < 0.2) {
    constraints.push(`⚠️ 局部同步序参量 R_local = ${R.toFixed(3)}（极低，邻居间也无同步）。Synchronization 评分必须 ≤ 2 分。`);
  } else if (R < 0.4) {
    constraints.push(`⚠️ 局部同步序参量 R_local = ${R.toFixed(3)}（低）。Synchronization 评分必须 ≤ 4 分。`);
  } else if (R > 0.85) {
    constraints.push(`ℹ️ 局部同步序参量 R_local = ${R.toFixed(3)}（极高）。局部几乎完全同步。`);
  } else if (R >= 0.4 && R <= 0.85) {
    constraints.push(`ℹ️ 局部同步序参量 R_local = ${R.toFixed(3)}（中等偏高）。存在有意义的局部同步。`);
  }

  // 簇数量约束
  const cc = stats.cluster_count;
  if (cc <= 1) {
    constraints.push(`ℹ️ 簇数量 = ${cc}。系统未形成多簇结构。`);
  }

  // 时序方差约束
  const tv = stats.temporal_variance;
  if (tv && tv.sync_R_variance > 0.05) {
    constraints.push(`⚠️ 同步序参量时序方差 = ${tv.sync_R_variance.toFixed(4)}（高）。系统不稳定，Structural Stability 必须 ≤ 4 分。`);
  }
  if (tv && tv.sync_R_trend === 'rising') {
    constraints.push(`ℹ️ 同步序参量呈上升趋势。系统可能正在自组织过程中，尚未收敛。`);
  }
  if (tv && tv.sync_R_trend === 'falling') {
    constraints.push(`ℹ️ 同步序参量呈下降趋势。系统可能正在解体。`);
  }

  if (constraints.length === 0) {
    return '## 硬数学指标\n所有指标处于正常范围，无强制约束。';
  }

  return `## 硬数学指标（必须遵守以下约束）\n${constraints.join('\n')}`;
}

/**
 * [RISK-4] 新颖性参考：展示最近几轮的描述，让 LLM 判断当前是否"打破既有分类"。
 */
function buildNoveltyReference(previousRuns) {
  if (previousRuns.length < 3) return '';

  const recent = previousRuns.slice(-5);
  return `## 历史实验描述（用于新颖性判断）
以下是最近 ${recent.length} 轮实验的涌现描述：
${recent.map(r => `- Run ${r.runId} (${r.score}分): ${r.description}`).join('\n')}

如果当前实验的视觉形态与以上所有描述都截然不同，请标记 is_novel=true。`;
}

function formatParams(params) {
  return Object.entries(params)
    .map(([k, v]) => `  - ${k}: ${v}`)
    .join('\n');
}

function formatStats(stats) {
  return `  - 平均速度: ${stats.avgSpeed?.toFixed(3)}
  - 速度分布: vx[${stats.velocitySpread?.minVx?.toFixed(2)}, ${stats.velocitySpread?.maxVx?.toFixed(2)}]
  - 已运行 tick 数: ${stats.tickCount}`;
}

function buildHistorySummary(previousRuns) {
  if (previousRuns.length === 0) return '';

  // 取最近的高分和低分各几条
  const sorted = [...previousRuns].sort((a, b) => b.score - a.score);
  const highs = sorted.slice(0, 3);
  const lows = sorted.slice(-2);

  let text = '## 历史实验参考\n';
  if (highs.length > 0) {
    text += '\n**高分实验（涌现较明显）：**\n';
    for (const run of highs) {
      text += `- Run ${run.runId}: 分数 ${run.score}, ${run.description}\n`;
      text += `  参数: temp=${run.params.temperature}, coupling=${run.params.couplingStrength}, radius=${run.params.perceptionRadius}\n`;
    }
  }
  if (lows.length > 0) {
    text += '\n**低分实验（涌现不明显）：**\n';
    for (const run of lows) {
      text += `- Run ${run.runId}: 分数 ${run.score}, ${run.description}\n`;
      text += `  参数: temp=${run.params.temperature}, coupling=${run.params.couplingStrength}, radius=${run.params.perceptionRadius}\n`;
    }
  }
  return text;
}
```

### 5.2 LLM API 调用

```javascript
// src/explorer/llm-evaluator.js (续)

/**
 * [优化] 接收单张拼图（Sprite Sheet），而非 5 张独立图片。
 * 拼图布局：左侧 2x2 时序切片（70%/80%/90%/100%），右侧 trail 大图。
 * 成本降低约 80%，且 LLM 做时序对比更直观。
 */
export async function evaluateScreenshot(
  compositeScreenshot,  // 单张拼图 base64
  params,
  stats,
  previousRuns,
  apiConfig
) {
  const prompt = buildEvalPrompt(params, stats, previousRuns);

  if (apiConfig.provider === 'openai') {
    return callOpenAI(prompt, compositeScreenshot, apiConfig);
  } else if (apiConfig.provider === 'anthropic') {
    return callAnthropic(prompt, compositeScreenshot, apiConfig);
  } else {
    throw new Error(`Unknown provider: ${apiConfig.provider}`);
  }
}

async function callOpenAI(prompt, imageBase64, config) {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model: config.model || 'gpt-4o',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            {
              type: 'image_url',
              image_url: {
                url: imageBase64.startsWith('data:')
                  ? imageBase64
                  : `data:image/png;base64,${imageBase64}`,
                detail: 'high',
              },
            },
          ],
        },
      ],
      max_tokens: 2500,
      temperature: 0.2,
    }),
  });

  const data = await response.json();
  return parseLLMResponse(data.choices[0].message.content);
}

async function callAnthropic(prompt, imageBase64, config) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': config.apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: config.model || 'claude-sonnet-4-20250514',
      max_tokens: 2500,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/png',
                data: imageBase64.replace(/^data:image\/png;base64,/, ''),
              },
            },
            { type: 'text', text: prompt },
          ],
        },
      ],
    }),
  });

  const data = await response.json();
  return parseLLMResponse(data.content[0].text);
}

function parseLLMResponse(text) {
  const jsonBlockMatch = text.match(/```json\s*([\s\S]*?)```/);
  if (jsonBlockMatch) {
    return JSON.parse(jsonBlockMatch[1].trim());
  }
  try {
    return JSON.parse(text);
  } catch {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start !== -1 && end !== -1) {
      return JSON.parse(text.slice(start, end + 1));
    }
    throw new Error('Failed to parse LLM response as JSON');
  }
}
```

---

## 六、数据存储与可溯源性

### 6.1 目录结构

```
emergent_system/
├── runs/                              # 所有实验数据
│   ├── manifest.jsonl                  # 索引文件（每行一条记录）
│   ├── 0001/
│   │   ├── params.json                # 完整参数配置
│   │   ├── config-snapshot.json       # 运行时引擎配置快照
│   │   ├── screenshot_t70.png         # [RISK-1] 70% 时刻瞬时截图（预热后）
│   │   ├── screenshot_t80.png         # [RISK-1] 80% 时刻瞬时截图
│   │   ├── screenshot_t90.png         # [RISK-1] 90% 时刻瞬时截图
│   │   ├── screenshot_t100.png        # [RISK-1] 100% 时刻瞬时截图
│   │   ├── screenshot_trail.png       # 60 帧累积轨迹图
│   │   ├── stats.json                 # [RISK-2] 硬数学指标 + 时序方差
│   │   └── llm-response.json          # LLM 完整评估（含 is_novel）
│   ├── 0002/
│   │   └── ...
│   └── ...
├── checkpoints/                       # 探索状态检查点
│   ├── exploration-state.json         # 当前探索进度（含 GA 代数）
│   └── best-results.json             # 高分结果索引
└── prompts/                           # 使用的 prompt 模板版本
    └── v2-eval-prompt.txt             # v2: 含硬约束 + 时序评估 + 新颖性
```

### 6.2 manifest.jsonl — 全局索引

每完成一轮实验，追加一行到 manifest.jsonl。这是一个 JSONL 文件（每行一个 JSON 对象），便于流式读取和追加：

```jsonl
{"runId":"0001","timestamp":"2026-06-23T14:30:00Z","params":{"temperature":0.35,"couplingStrength":1.2,"perceptionRadius":60},"composite_score":6.2,"emergence_level":"moderate","description":"出现了小规模粒子簇，有微弱的同步倾向","prompt_version":"v1","llm_model":"gpt-4o","strategy":"random"}
{"runId":"0002","timestamp":"2026-06-23T14:35:00Z","params":{"temperature":0.15,"couplingStrength":3.0,"perceptionRadius":80},"composite_score":8.1,"emergence_level":"strong","description":"壮观的同步旋涡结构，多个旋转臂清晰可辨","prompt_version":"v1","llm_model":"gpt-4o","strategy":"llm-guided"}
```

### 6.3 llm-response.json — LLM 原始反馈

完整保存 LLM 的返回，包括所有分数和建议：

```json
{
  "runId": "0002",
  "timestamp": "2026-06-23T14:35:30Z",
  "model": "gpt-4o",
  "prompt_version": "v1",
  "raw_response": "...原始文本...",
  "parsed": {
    "scores": {
      "spatial_clustering": 8,
      "structural_stability": 7,
      "synchronization": 9,
      "hierarchical_complexity": 6,
      "aesthetic_emergence": 8
    },
    "composite_score": 7.75,
    "emergence_level": "strong",
    "description": "壮观的同步旋涡结构，多个旋转臂清晰可辨",
    "observations": [
      "粒子明显聚集成 3-4 个大的旋转簇",
      "簇与簇之间有粒子流连接",
      "整体呈现出类似星系旋臂的结构"
    ],
    "is_false_positive": false,
    "false_positive_reason": null,
    "suggestion": {
      "direction": "增大感知半径，看是否能形成更大的同步结构",
      "param_hints": {
        "perceptionRadius": "增大",
        "couplingStrength": "保持",
        "temperature": "略微减小以增强稳定性"
      }
    }
  },
  "latency_ms": 3200,
  "token_usage": {
    "input": 1523,
    "output": 487
  }
}
```

### 6.4 exploration-state.json — 探索进度

记录探索器的当前状态，支持断点续跑：

```json
{
  "experimentName": "kuramoto-phase1",
  "startedAt": "2026-06-23T14:00:00Z",
  "lastRunId": 23,
  "totalRuns": 23,
  "strategy": "llm-guided",
  "bestScore": 8.4,
  "bestRunId": 15,
  "paramSpace": "PARAM_SPACE v1",
  "strategyState": {
    "type": "llm-guided",
    "explored": [
      {"temperature": [0.1, 0.5], "couplingStrength": [0.5, 3.0]},
      {"temperature": [0.2, 0.4], "couplingStrength": [2.0, 4.0]}
    ],
    "focusRegion": {
      "temperature": [0.2, 0.35],
      "couplingStrength": [1.5, 3.0],
      "reason": "历史数据表明此区域涌现最明显"
    }
  },
  "budget": {
    "maxRuns": 100,
    "maxLLMCalls": 100,
    "runsUsed": 23,
    "llmCallsUsed": 23
  }
}
```

---

## 七、参数探索策略

### 7.1 策略一：随机采样（Baseline）

最简单的策略，用于建立初始基线数据：

```javascript
// src/explorer/strategies/random.js

export class RandomStrategy {
  constructor(paramSpace, rng) {
    this.paramSpace = paramSpace;
    this.rng = rng;
  }

  next() {
    const params = {};
    for (const [key, spec] of Object.entries(this.paramSpace)) {
      if (spec.category === 'render') {
        // 渲染参数使用默认值，不参与探索
        params[key] = spec.step ? (spec.min + spec.max) / 2 : spec.min;
        continue;
      }
      const range = spec.max - spec.min;
      const steps = Math.floor(range / spec.step);
      const randomStep = Math.floor(this.rng() * (steps + 1));
      params[key] = spec.min + randomStep * spec.step;
      // 四舍五入到 step 精度
      params[key] = Math.round(params[key] / spec.step) * spec.step;
    }
    return params;
  }
}
```

### 7.2 策略二：LLM 引导搜索（核心策略）

让 LLM 根据历史记录建议下一轮参数。**[RISK-3 修复]** prompt 中加入了相变边界探测提示。

```javascript
// src/explorer/strategies/llm-guided.js

export class LLMGuidedStrategy {
  constructor(paramSpace, apiConfig) {
    this.paramSpace = paramSpace;
    this.apiConfig = apiConfig;
  }

  async next(previousRuns) {
    const prompt = this.buildGuidancePrompt(previousRuns);
    const response = await this.callLLM(prompt);
    return this.parseParamSuggestion(response);
  }

  buildGuidancePrompt(previousRuns) {
    const recent = previousRuns.slice(-10);
    const best = [...previousRuns].sort((a, b) => b.score - a.score).slice(0, 3);

    // [RISK-3] 检测可能的相变边界
    const phaseBoundaryWarning = detectPhaseBoundaries(recent);

    return `你是一个复杂系统实验科学家。你正在调整一个人工粒子宇宙的参数，寻找涌现现象。

## 参数空间
${JSON.stringify(this.paramSpace, null, 2)}

## 最近 ${recent.length} 轮实验结果
${recent.map(r =>
  `- Run ${r.runId}: score=${r.score}, level=${r.level}
   params: ${JSON.stringify(r.params)}
   LLM评语: ${r.description}`
).join('\n')}

## 历史最佳结果
${best.map(r =>
  `- Run ${r.runId}: score=${r.score}, ${r.description}
   params: ${JSON.stringify(r.params)}`
).join('\n')}

${phaseBoundaryWarning}

## 你的任务
基于以上实验记录，建议下一轮实验的参数。目标是找到更强的涌现现象。

请输出一个 JSON 对象：

\`\`\`json
{
  "reasoning": "<你的分析思路，100字以内>",
  "params": {
    "temperature": <值>,
    "couplingStrength": <值>,
    "perceptionRadius": <值>,
    "decayRate": <值>,
    "particleCount": <值>,
    "initialSpeed": <值>,
    "friction": <值>,
    "maxSpeed": <值>,
    "trailAlpha": <值>,
    "particleSize": <值>
  },
  "hypothesis": "<你期望看到什么现象，50字以内>"
}
\`\`\`

[CRITICAL] 非线性系统警告：
1. 涌现系统存在"相变悬崖"——参数微小变化可能导致结果剧变
2. 如果两组相近参数的得分差异 > 3 分，你可能找到了相变边界
3. 发现相变边界时：围绕边界做极小步长采样（step 的 1/4），不要向外扩展
4. 所有参数必须在定义的 min/max 范围内
5. 不要重复已试过的参数组合`;
  }

  async callLLM(prompt) {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiConfig.apiKey}`,
      },
      body: JSON.stringify({
        model: this.apiConfig.model || 'gpt-4o',
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 1000,
        temperature: 0.7,
      }),
    });
    const data = await response.json();
    return data.choices[0].message.content;
  }

  parseParamSuggestion(text) {
    const jsonBlockMatch = text.match(/```json\s*([\s\S]*?)```/);
    if (jsonBlockMatch) return JSON.parse(jsonBlockMatch[1].trim());
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start !== -1 && end !== -1) return JSON.parse(text.slice(start, end + 1));
    throw new Error('Failed to parse LLM parameter suggestion');
  }
}

/**
 * [RISK-3] 相变边界检测。
 * 如果最近几轮中，参数相近但分数差异大，说明可能在相变边界附近。
 */
function detectPhaseBoundaries(runs) {
  if (runs.length < 3) return '';

  const boundaries = [];
  for (let i = 1; i < runs.length; i++) {
    const prev = runs[i - 1];
    const curr = runs[i];
    const scoreDiff = Math.abs(curr.score - prev.score);

    if (scoreDiff > 3) {
      // 计算参数空间中的距离
      const paramDist = computeParamDistance(prev.params, curr.params);
      if (paramDist < 0.3) {  // 参数相近但分数差异大
        boundaries.push({
          run1: prev.runId,
          run2: curr.runId,
          scoreDiff,
          params1: prev.params,
          params2: curr.params,
        });
      }
    }
  }

  if (boundaries.length === 0) return '';

  return `## ⚠️ 检测到可能的相变边界
以下实验对的参数相近但得分差异巨大，说明你在相变边界附近：
${boundaries.map(b =>
  `- Run ${b.run1} (${b.scoreDiff.toFixed(1)}分差异): ` +
  `temp ${b.params1.temperature}→${b.params2.temperature}, ` +
  `coupling ${b.params1.couplingStrength}→${b.params2.couplingStrength}`
).join('\n')}

建议：在这些参数值之间做极小步长的精细采样，找到精确的相变临界点。`;
}

function computeParamDistance(p1, p2) {
  // 归一化欧氏距离（各参数归一化到 [0,1]）
  const keys = ['temperature', 'couplingStrength', 'perceptionRadius'];
  let sum = 0;
  for (const k of keys) {
    const v1 = p1[k] ?? 0;
    const v2 = p2[k] ?? 0;
    const diff = (v1 - v2) / (Math.abs(v1) + Math.abs(v2) + 0.001);
    sum += diff * diff;
  }
  return Math.sqrt(sum / keys.length);
}
```

### 7.3 策略三：遗传算法精细搜索 [RISK-3 修复]

当 LLM 引导搜索锁定一个"有趣区域"后，用遗传算法做精细搜索。
遗传算法的优势：通过变异和交叉探索参数空间，天然适配非线性相变边界。

```javascript
// src/explorer/strategies/genetic.js

export class GeneticStrategy {
  constructor(paramSpace, rng, config = {}) {
    this.paramSpace = paramSpace;
    this.rng = rng;
    this.populationSize = config.populationSize || 10;
    this.mutationRate = config.mutationRate || 0.15;    // 每个参数的变异概率
    this.mutationScale = config.mutationScale || 0.1;   // 变异幅度（相对于参数范围）
    this.eliteCount = config.eliteCount || 3;           // 保留前 N 名直接进入下一代
    this.population = [];
    this.generation = 0;
  }

  /**
   * 从高分结果初始化种群。
   * @param {Array} seedRuns - 历史高分实验 [{params, score}, ...]
   */
  initialize(seedRuns = []) {
    this.population = [];

    // 用高分结果作为种子
    for (const run of seedRuns.slice(0, this.eliteCount)) {
      this.population.push({ params: { ...run.params }, score: run.score });
    }

    // 补充随机个体
    while (this.population.length < this.populationSize) {
      this.population.push({ params: this.randomParams(), score: 0 });
    }

    this.generation = 0;
  }

  /**
   * 根据上一代的评分，生成下一代。
   * 包含精英衰减：精英的分数乘以 0.95，防止历史偶然高分永远霸榜。
   */
  evolve(evaluatedPopulation) {
    // 按分数排序
    evaluatedPopulation.sort((a, b) => b.score - a.score);

    const nextGen = [];

    // 精英保留：前 N 名直接进入下一代，但分数衰减 5%
    // 迫使精英必须在下一代被重新评估验证
    for (let i = 0; i < this.eliteCount && i < evaluatedPopulation.length; i++) {
      const elite = { params: { ...evaluatedPopulation[i].params }, score: 0 };
      elite._prevScore = evaluatedPopulation[i].score * 0.95; // 衰减标记，供日志用
      nextGen.push(elite);
    }

    // 交叉 + 变异产生剩余个体
    while (nextGen.length < this.populationSize) {
      const parent = this.tournamentSelect(evaluatedPopulation);
      const child = this.mutate({ ...parent.params });
      nextGen.push({ params: child, score: 0 });
    }

    this.population = nextGen;
    this.generation++;
    return this.population;
  }

  /**
   * 获取下一个待评估的个体。
   */
  next() {
    // 返回种群中第一个 score=0 的个体
    const individual = this.population.find(p => p.score === 0);
    return individual ? individual.params : null;
  }

  /** 锦标赛选择：随机选 3 个，取最高分 */
  tournamentSelect(pop) {
    let best = null;
    for (let i = 0; i < 3; i++) {
      const idx = Math.floor(this.rng() * pop.length);
      if (!best || pop[idx].score > best.score) {
        best = pop[idx];
      }
    }
    return best;
  }

  /** 变异：对每个参数以一定概率做微小扰动 */
  mutate(params) {
    for (const [key, spec] of Object.entries(this.paramSpace)) {
      if (spec.category === 'render') continue;
      if (this.rng() < this.mutationRate) {
        const range = spec.max - spec.min;
        const delta = (this.rng() - 0.5) * 2 * range * this.mutationScale;
        params[key] = Math.max(spec.min, Math.min(spec.max, params[key] + delta));
        // 对齐到 step
        params[key] = Math.round(params[key] / spec.step) * spec.step;
      }
    }
    return params;
  }

  randomParams() {
    const params = {};
    for (const [key, spec] of Object.entries(this.paramSpace)) {
      if (spec.category === 'render') {
        params[key] = (spec.min + spec.max) / 2;
        continue;
      }
      const steps = Math.floor((spec.max - spec.min) / spec.step);
      params[key] = spec.min + Math.floor(this.rng() * (steps + 1)) * spec.step;
    }
    return params;
  }
}
```

### 7.4 三阶段探索流程

```
Phase A: 随机采样（20 轮）
  └─ 建立粗略的参数-分数映射基线

Phase B: LLM 引导搜索（30 轮）
  └─ LLM 识别"有趣区域"，同时检测相变边界

Phase C: 遗传算法精细搜索（50 轮）
  └─ 在 LLM 锁定的区域里，用 GA 做精细变异搜索
  └─ 天然适配非线性参数空间，不会被"悬崖"卡住
```

---

## 八、主控制器（Orchestrator）

将所有组件串联起来：

```javascript
// src/explorer/orchestrator.js

import fs from 'fs';
import path from 'path';
import { createRNG } from '../prng.js';
import { PARAM_SPACE } from './param-space.js';
import { RandomStrategy } from './strategies/random.js';
import { LLMGuidedStrategy } from './strategies/llm-guided.js';
import { GeneticStrategy } from './strategies/genetic.js';
import { runExperiment } from './runner.js';
import { evaluateScreenshots } from './llm-evaluator.js';

const RUNS_DIR = path.resolve('runs');
const CHECKPOINT_DIR = path.resolve('checkpoints');

export class Orchestrator {
  constructor(config) {
    this.config = config;
    this.rng = createRNG(config.seed || 42);
    this.state = this.loadOrCreateState();
    this.previousRuns = this.loadAllRuns();
    this.geneticStrategy = null;
  }

  /**
   * 三阶段探索：random → llm-guided → genetic
   */
  async runAll() {
    console.log('=== Phase A: Random Sampling (20 runs) ===');
    this.state.strategy = 'random';
    await this.run(20);

    console.log('\n=== Phase B: LLM-Guided Search (30 runs) ===');
    this.state.strategy = 'llm-guided';
    await this.run(30);

    console.log('\n=== Phase C: Genetic Fine-Search (50 runs) ===');
    this.state.strategy = 'genetic';
    await this.run(50);

    this.printSummary();
  }

  async run(maxRounds = 50) {
    console.log(`[Orchestrator] Strategy: ${this.state.strategy}, Budget: ${this.state.budget.maxRuns - this.state.budget.runsUsed} remaining`);

    const strategy = this.createStrategy();

    for (let round = 0; round < maxRounds; round++) {
      if (this.state.budget.runsUsed >= this.state.budget.maxRuns) {
        console.log('[Orchestrator] Budget exhausted.');
        break;
      }

      const roundId = this.state.lastRunId + 1;
      console.log(`\n--- Round ${roundId} (${this.state.strategy}) ---`);

      // 1. 获取下一组参数
      let params;
      let suggestion = null;

      if (this.state.strategy === 'llm-guided' && this.previousRuns.length >= 3) {
        suggestion = await strategy.next(this.previousRuns);
        params = suggestion.params;
        console.log(`[LLM] Reasoning: ${suggestion.reasoning}`);
      } else if (this.state.strategy === 'genetic') {
        params = strategy.next();
        if (!params) {
          // 当前代评估完毕，进化到下一代
          console.log(`[GA] Generation ${this.geneticStrategy.generation} complete. Evolving...`);
          this.geneticStrategy.evolve(this.geneticStrategy.population);
          params = strategy.next();
        }
        console.log(`[GA] Gen ${this.geneticStrategy.generation}, individual in population`);
      } else {
        params = strategy.next();
        console.log(`[Random] Params: temp=${params.temperature?.toFixed(2)}, coupling=${params.couplingStrength?.toFixed(2)}`);
      }

      if (!params) break;

      // 2. 运行仿真（返回拼图 + 原始切片 + 统计）
      console.log('[Runner] Running simulation...');
      const { runId, runDir, composite_screenshot, stats } = await runExperiment(params, roundId);
      console.log(`[Runner] R_local=${stats.sync_order_parameter?.toFixed(3)}, Clusters=${stats.cluster_count}`);

      // 3. LLM 评估（单张拼图 + 硬数学约束）
      console.log('[LLM] Evaluating (1 composite image + math constraints)...');
      const evalResult = await evaluateScreenshot(
        composite_screenshot, params, stats, this.previousRuns, this.config.llm
      );
      console.log(`[LLM] Score: ${evalResult.composite_score}, Level: ${evalResult.emergence_level}`);
      console.log(`[LLM] Description: ${evalResult.description}`);
      if (evalResult.is_novel) {
        console.log(`[LLM] ★ NOVEL: ${evalResult.novelty_note}`);
      }

      // 4. 保存 LLM 反馈
      fs.writeFileSync(
        path.join(runDir, 'llm-response.json'),
        JSON.stringify({
          runId, timestamp: new Date().toISOString(),
          model: this.config.llm.model, prompt_version: 'v2',
          parsed: evalResult,
        }, null, 2)
      );

      // 5. 追加到 manifest
      fs.appendFileSync(
        path.join(RUNS_DIR, 'manifest.jsonl'),
        JSON.stringify({
          runId, timestamp: new Date().toISOString(),
          params,
          composite_score: evalResult.composite_score,
          emergence_level: evalResult.emergence_level,
          description: evalResult.description,
          is_novel: evalResult.is_novel || false,
          sync_R: stats.sync_order_parameter,
          prompt_version: 'v2',
          llm_model: this.config.llm.model,
          strategy: this.state.strategy,
        }) + '\n'
      );

      // 6. 更新状态
      const runRecord = {
        runId, params,
        score: evalResult.composite_score,
        level: evalResult.emergence_level,
        description: evalResult.description,
      };
      this.previousRuns.push(runRecord);

      // 遗传算法：回写分数到种群
      if (this.state.strategy === 'genetic' && this.geneticStrategy) {
        const individual = this.geneticStrategy.population.find(p => p.score === 0);
        if (individual) individual.score = evalResult.composite_score;
      }

      this.state.lastRunId = roundId;
      this.state.budget.runsUsed++;
      this.state.budget.llmCallsUsed++;
      if (evalResult.composite_score > this.state.bestScore) {
        this.state.bestScore = evalResult.composite_score;
        this.state.bestRunId = roundId;
      }
      this.saveState();

      if (evalResult.composite_score >= 7.0) {
        this.recordBestResult(runId, evalResult, params, stats);
        console.log(`[!] HIGH SCORE: Run ${runId} scored ${evalResult.composite_score}!`);
      }
    }
  }

  createStrategy() {
    switch (this.state.strategy) {
      case 'random':
        return new RandomStrategy(PARAM_SPACE, this.rng);
      case 'llm-guided':
        return new LLMGuidedStrategy(PARAM_SPACE, this.config.llm);
      case 'genetic':
        if (!this.geneticStrategy) {
          this.geneticStrategy = new GeneticStrategy(PARAM_SPACE, this.rng);
          // 用历史高分结果初始化种群
          const seeds = [...this.previousRuns]
            .sort((a, b) => b.score - a.score)
            .slice(0, 5);
          this.geneticStrategy.initialize(seeds);
        }
        return this.geneticStrategy;
      default:
        throw new Error(`Unknown strategy: ${this.state.strategy}`);
    }
  }

  recordBestResult(runId, evalResult, params, stats) {
    const bestPath = path.join(CHECKPOINT_DIR, 'best-results.json');
    let best = [];
    if (fs.existsSync(bestPath)) {
      best = JSON.parse(fs.readFileSync(bestPath, 'utf-8'));
    }
    best.push({
      runId, score: evalResult.composite_score,
      level: evalResult.emergence_level,
      description: evalResult.description,
      is_novel: evalResult.is_novel || false,
      params, stats,
      timestamp: new Date().toISOString(),
    });
    best.sort((a, b) => b.score - a.score);
    fs.writeFileSync(bestPath, JSON.stringify(best, null, 2));
  }

  loadOrCreateState() {
    const statePath = path.join(CHECKPOINT_DIR, 'exploration-state.json');
    if (fs.existsSync(statePath)) {
      return JSON.parse(fs.readFileSync(statePath, 'utf-8'));
    }
    const state = {
      experimentName: `exp-${Date.now()}`,
      startedAt: new Date().toISOString(),
      lastRunId: 0, totalRuns: 0,
      strategy: 'random',
      bestScore: 0, bestRunId: 0,
      budget: { maxRuns: 100, maxLLMCalls: 100, runsUsed: 0, llmCallsUsed: 0 },
    };
    fs.mkdirSync(CHECKPOINT_DIR, { recursive: true });
    fs.writeFileSync(statePath, JSON.stringify(state, null, 2));
    return state;
  }

  loadAllRuns() {
    const manifestPath = path.join(RUNS_DIR, 'manifest.jsonl');
    if (!fs.existsSync(manifestPath)) return [];
    return fs.readFileSync(manifestPath, 'utf-8')
      .trim().split('\n').filter(Boolean)
      .map(line => {
        const e = JSON.parse(line);
        return {
          runId: e.runId, params: e.params,
          score: e.composite_score, level: e.emergence_level,
          description: e.description,
        };
      });
  }

  saveState() {
    fs.writeFileSync(
      path.join(CHECKPOINT_DIR, 'exploration-state.json'),
      JSON.stringify(this.state, null, 2)
    );
  }

  printSummary() {
    console.log('\n========== EXPERIMENT SUMMARY ==========');
    console.log(`Total runs: ${this.state.budget.runsUsed}`);
    console.log(`Best score: ${this.state.bestScore} (Run ${this.state.bestRunId})`);

    const novelRuns = this.previousRuns.filter(r => r.is_novel);
    if (novelRuns.length > 0) {
      console.log(`\n★ Novel discoveries: ${novelRuns.length}`);
      for (const r of novelRuns) {
        console.log(`  Run ${r.runId}: ${r.score} - ${r.description}`);
      }
    }

    const bestRuns = [...this.previousRuns].sort((a, b) => b.score - a.score).slice(0, 5);
    console.log('\nTop 5 results:');
    for (const run of bestRuns) {
      console.log(`  Run ${run.runId}: ${run.score} - ${run.description}`);
    }
    console.log('========================================');
  }
}
```

---

## 九、使用方式

### 9.1 启动脚本

```javascript
// scripts/start-exploration.js

import { Orchestrator } from '../src/explorer/orchestrator.js';

const orchestrator = new Orchestrator({
  seed: 42,
  llm: {
    provider: 'openai',      // 或 'anthropic'
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4o',
  },
});

// 三阶段探索：random(20) → llm-guided(30) → genetic(50)
orchestrator.state.budget.maxRuns = 100;
await orchestrator.runAll();
```

### 9.2 运行命令

```bash
# 安装依赖
npm install puppeteer

# 启动 Vite dev server（一个终端）
npm run dev

# 启动探索（另一个终端）
node scripts/start-exploration.js
```

### 9.3 查看结果

```bash
# 查看全局索引
cat runs/manifest.jsonl | jq .

# 查看最佳结果
cat checkpoints/best-results.json | jq .

# 查看某一轮的详细信息
cat runs/0015/llm-response.json | jq .parsed

# 打开某一轮的截图
open runs/0015/screenshot.png
```

---

## 十、成本估算

| 项目 | 每轮消耗 | 100 轮总计 |
|---|---|---|
| LLM 评估调用（1 张拼图 + 硬约束 prompt） | ~2500 input tokens + ~600 output tokens (GPT-4o) | ~$4 |
| LLM 引导调用（仅 Phase B，30 轮） | ~3000 input tokens + ~500 output tokens | ~$1.50 |
| Puppeteer 内存 | ~200MB / 实例 | 串行运行，峰值 200MB |
| 磁盘存储 | ~3MB/轮（5 张原始切片 + 1 张拼图 + JSON） | ~300MB |
| 单轮耗时 | ~10 秒（仿真） + ~3 秒（LLM，1 图） | ~22 分钟/100 轮 |

**总预算建议**：100 轮 ≈ $6 的 LLM API 费用，22 分钟时间。

> 注：拼图策略将每轮图片 token 从 5 张降至 1 张，成本下降约 70%。拼图同时提高了 LLM 的时序对比准确率——所有信息在一张图里，无需跨图记忆。

---

## 十一、可溯源性保证

本设计在以下环节保证完全可溯源：

| 环节 | 存储内容 | 文件位置 |
|---|---|---|
| 参数输入 | 完整参数 JSON | `runs/<id>/params.json` |
| 仿真配置 | 推导出的引擎配置 | `runs/<id>/config-snapshot.json` |
| 时序截图 | 4 张时间切片 + 1 张 trail | `runs/<id>/screenshot_t40.png` ... `trail.png` |
| 物理统计 | 硬数学指标 + 时序方差 | `runs/<id>/stats.json` |
| LLM 评估 | 原始返回 + 解析结果 + 新颖性标记 | `runs/<id>/llm-response.json` |
| Prompt 版本 | 使用的 prompt 模板 | `prompts/v2-eval-prompt.txt` |
| 全局索引 | 所有轮次的摘要（含 sync_R, is_novel） | `runs/manifest.jsonl` |
| 探索状态 | 策略、预算、进度、GA 代数 | `checkpoints/exploration-state.json` |
| 高分记录 | 最佳结果索引（含物理统计） | `checkpoints/best-results.json` |

任何时候拿到 `runs/<id>/params.json`，都可以精确复现该轮实验（种子固定为 42）。

### 新增的可溯源维度（v2）

| 维度 | 存储位置 | 用途 |
|---|---|---|
| 同步序参量 R | `stats.json → sync_order_parameter` | 防伪：数学锚定涌现程度 |
| 时序方差 | `stats.json → temporal_variance` | 防伪：判断结构是否稳定 |
| 新颖性标记 | `llm-response.json → is_novel` | 发现未知涌现形态 |
| 相变边界检测 | LLM 引导 prompt 中动态计算 | 引导精细搜索 |
| 遗传算法代数 | `exploration-state.json → generation` | 追踪 GA 搜索进度 |

---

## 附录：v1 → v2 变更摘要

v2 的所有修改围绕 4 个已知风险展开。以下是防线到代码的完整映射：

### 风险一：时间坍缩 → 多帧时序切片 + 时序方差

| 修改 | 位置 | 效果 |
|---|---|---|
| 仿真运行器在 40%/60%/80%/100% 处各截一张瞬时图 | `runner.js` → `slicePoints` | LLM 可对比 4 张图判断结构是否稳定 |
| 统计系统在最后 200 帧每 10 帧采样 | `runner.js` → `timeSeries` | 计算 sync_R/avgSpeed/clusterCount 的方差 |
| 时序方差写入 stats | `runner.js` → `temporal_variance` | LLM 有硬数字判断稳定性 |
| 评估 prompt 要求对比 4 张切片 | `llm-evaluator.js` → "时序评估要求" | LLM 不能只看 trail 图打分 |
| 稳定性维度的评分标准改为基于时序切片 | `llm-evaluator.js` → Structural Stability rubric | 有明确的"4 张切片一致"标准 |

### 风险二：空想性错视 → 同步序参量 R + 硬数学约束

| 修改 | 位置 | 效果 |
|---|---|---|
| 引擎计算 Kuramoto 同步序参量 R | `runner.js` → `computeSyncR()` | R ≈ 0 混沌, R ≈ 1 同步 |
| R 和簇数量写入 stats | `runner.js` → `sync_order_parameter`, `cluster_count` | LLM 拿到客观物理量 |
| prompt 中根据 R 值强制限制评分 | `llm-evaluator.js` → `buildHardConstraints()` | R < 0.15 时 Synchronization 最高 2 分 |
| 时序方差大时强制限制稳定性评分 | `buildHardConstraints()` | 方差高 → Stability 最高 4 分 |

### 风险三：相变悬崖 → 遗传算法 + 相变边界探测

| 修改 | 位置 | 效果 |
|---|---|---|
| 新增 GeneticStrategy | `strategies/genetic.js` | 通过变异+交叉探索非线性空间 |
| 三阶段探索流程 | `orchestrator.js` → `runAll()` | random → LLM引导 → GA精细搜索 |
| 相变边界自动检测 | `strategies/llm-guided.js` → `detectPhaseBoundaries()` | 参数相近但分数差异大 → 标记为相变边界 |
| LLM 引导 prompt 加入非线性警告 | `llm-guided.js` → buildGuidancePrompt | LLM 知道不能线性外推 |
| 发现相变边界时建议精细采样 | prompt 中的 "[CRITICAL] 非线性系统警告" | 围绕边界做 1/4 step 的微调 |

### 风险四：模式匹配退化 → 新颖性维度 + 异常标记

| 修改 | 位置 | 效果 |
|---|---|---|
| 评分维度从 5 个增加到 6 个 | `llm-evaluator.js` → 新增 "novelty" 维度 | 新颖性占权重 20% |
| prompt 展示最近 5 轮描述作为对比基线 | `buildNoveltyReference()` | LLM 能判断当前是否"打破既有分类" |
| LLM 输出新增 `is_novel` 和 `novelty_note` | eval prompt JSON schema | 新发现会被显式标记 |
| manifest.jsonl 记录 `is_novel` | `orchestrator.js` | 全局索引可直接筛选新颖发现 |
| printSummary 展示新颖发现 | `orchestrator.js` → `printSummary()` | 实验结束时高亮展示 |
| composite_score 公式调整 | eval prompt | novelty 权重 0.2，与 spatial 并列 |

### 加权公式变更

```
v1: composite = spatial×0.2 + stability×0.2 + sync×0.25 + hierarchy×0.15 + aesthetic×0.2
v2: composite = spatial×0.15 + stability×0.2 + sync×0.2 + hierarchy×0.1 + aesthetic×0.15 + novelty×0.2
```

新增的 novelty 维度分走了其他维度的权重，确保"意料之外的涌现"不会被"不符合预设标准"而压低分数。

---

## 附录 B：v2 → v3 变更摘要

v3 的修改围绕 1 个致命物理学漏洞和 3 个工程优化展开。

### 致命修复：全局 R → 局部平均 R_local

| 修改 | 位置 | 效果 |
|---|---|---|
| `computeSyncR()` 改为计算局部序参量 | `runner.js` | 每个粒子只与感知半径内邻居计算同步度，再求全图平均 |
| 多簇涌现不再被误杀 | `computeSyncR()` | 3 个各自同步但互不同步的簇 → R_local ≈ 1（而非全局 R = 0） |
| 硬约束阈值调整 | `buildHardConstraints()` | R < 0.2 才触发低分约束（局部 R 天然比全局 R 高） |
| 函数签名变化 | `computeSyncR(state, spatialHash, radius)` | 需要传入空间索引和感知半径 |

**为什么这是致命的**：全局 R 会把"3 个各自同步的簇"判为 R=0，触发硬约束强制压分。这恰好是最美的涌现形态之一。

### 优化 1：时序切片窗口后移

| 修改 | 位置 | 效果 |
|---|---|---|
| 切片点从 40/60/80/100% 改为 70/80/90/100% | `runner.js` → `slicePoints` | 前 70% 作为预热期，只在稳定期考察 |
| 文件名从 `t40` 改为 `t70` | Puppeteer 脚本 | 保持命名一致 |

### 优化 2：拼图（Sprite Sheet）

| 修改 | 位置 | 效果 |
|---|---|---|
| 浏览器端将 5 张图拼成 1 张全景图 | `runner.js` → composite canvas | 左侧 2×2 时序切片 + 右侧 trail 大图 |
| LLM 只接收 1 张拼图 | `llm-evaluator.js` | 成本降低 ~70%，时序对比更直观 |
| 原始 5 张切片仍独立保存 | Puppeteer 脚本 | 存档/人工查看不受影响 |
| 成本估算更新 | 第十章 | 100 轮从 ~$12 降至 ~$6 |

### 优化 3：GA 精英衰减

| 修改 | 位置 | 效果 |
|---|---|---|
| 精英分数乘以 0.95 衰减 | `genetic.js` → `evolve()` | 防止 LLM 幻觉导致的偶然高分永远霸榜 |
| 衰减后 score 重置为 0 | `evolve()` | 精英必须在下一代被重新评估 |

### v3 综合评分公式（不变）

```
composite = spatial×0.15 + stability×0.2 + sync×0.2 + hierarchy×0.1 + aesthetic×0.15 + novelty×0.2
```

### 成本对比

| 版本 | 每轮图片 Token | 100 轮 LLM 成本 | 单轮耗时 |
|---|---|---|---|
| v1 | ~2000 (1 张) | ~$2.50 | ~15 秒 |
| v2 | ~8000 (5 张) | ~$12 | ~18 秒 |
| **v3** | ~2500 (1 张拼图) | **~$6** | ~13 秒 |

v3 在保持 v2 全部防线的同时，成本比 v2 降低 50%，且拼图提高了 LLM 的判断准确率。
