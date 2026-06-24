/**
 * LLM Evaluator — sends ecosystem simulation screenshots to a multimodal LLM
 * for emergence assessment.
 *
 * Evaluates multi-species slime mold ecosystems with nutrient dynamics.
 */

/**
 * Build the evaluation prompt for ecosystem experiments.
 */
export function buildEvalPrompt(params, stats, previousRuns = []) {
  const historySection = buildHistorySummary(previousRuns);
  const hardConstraints = buildHardConstraints(stats);
  const noveltyReference = buildNoveltyReference(previousRuns);

  return `你是一个复杂生态系统研究专家，正在评估一个人工多物种粘菌生态系统的涌现现象。

## 实验背景
这是一个多物种粘菌（Physarum）生态系统模拟，运行了 ${stats.tickCount || 15000} tick。包含 3 个物种：
- **Explorer**（橙金色）：快速、广感知、高代谢消耗，擅长发现新营养源
- **Grazer**（绿色）：慢速、低代谢、高效利用，能在贫瘠区域存活
- **Harvester**（青蓝色）：平衡型，每单位营养提取量最高

系统特性：
- **废弃物/毒素场**：个体进食时产生废弃物，高废弃物区域会增加代谢消耗并排斥个体
- **移动营养热点**：营养注入中心缓慢漂移，创造"季节变化"，迫使生态网络迁移
- **休眠孢子机制**：能量耗尽时进入休眠而非死亡，附近有营养时苏醒
- **跨物种信息素**：物种间通过其他物种的信息素产生吸引/排斥，形成领地结构

系统参数：
${formatParams(params)}

运行统计：
${formatStats(stats)}

${historySection}

${hardConstraints}

## 你的任务

你收到了一张拼图（Sprite Sheet），布局如下：
- 左侧（2×2 网格）：仿真运行到 70%、80%、90%、100% 时刻的截图（全部在预热期之后），每个带有时间戳标注
- 右上：最终 80 帧的累积轨迹图
- 右下：差异图（Δ Map）— 蓝色=稳定区域，红色=变化区域。这是 70% 和 100% 时刻的像素差分

差异图解读：
- 大面积蓝色 → 系统已稳定，结构固化
- 大面积红色 → 系统仍在剧烈变化
- 蓝底红色纹理 → 稳定结构上有微流动（最有趣的动态复杂度）

请评估以下维度（每项 1-10 分）：

**1. 物种共存 (Species Coexistence)**
- 1分：一个物种完全主导，其他几近灭绝
- 5分：三个物种都存活但数量不均
- 10分：三个物种稳定共存，比例接近均衡

**2. 空间自组织 (Spatial Self-Organization)**
- 1分：所有物种均匀随机分布
- 5分：可见的领地划分或聚集模式
- 10分：壮观的多层级空间结构，物种间形成复杂的领地边界

**3. 营养网络 (Nutrient Network)**
- 1分：营养随机分散，无结构
- 5分：可见的营养通道或网络雏形
- 10分：有机体与营养场形成类似血管网络的自组织结构

**4. 结构稳定性 (Structural Stability)** ← 参考时序切片（70%-100%，已过预热期）
- 1分：四张切片完全不同，结构瞬息万变
- 5分：主要结构在至少 3 张切片中持续存在
- 10分：结构在 70%-100% 整个窗口内完全稳定

**5. 动态复杂度 (Dynamic Complexity)** ← 参考差异图
- 1分：简单扩散或静止，差异图全蓝
- 5分：可见的流动、脉动或周期行为，差异图有局部红色纹理
- 10分：令人惊叹的多尺度动力学——扩张收缩循环、螺旋波、追逐营养热点的宏观迁移。差异图显示大范围结构性变化

**6. 美学涌现 (Aesthetic Emergence)**
- 1分：视觉噪声
- 5分：有吸引力的图案
- 10分：让人屏息的有机生态景观

### 废弃物/毒素系统评估
请注意画面中的紫红色区域（废弃物/毒素场）：
- 如果废弃物形成了清晰的"禁区"或"毒素环"，而物种围绕这些区域形成领地 → 高分
- 如果废弃物场几乎不可见 → 说明系统没有产生有意义的废物循环
- 如果看到"脉冲式扩张"（废弃物积累→逃离→新区域开拓）→ Dynamic Complexity 高分

### 负面清单（请扣分）
- 营养场完全空白 → 最高 2 分
- 所有个体堆积在角落 → 最高 2 分
- 物种颜色完全混在一起无法区分 → 结构性扣分
- 数值溢出导致的异常图案 → 最高 1 分
- 系统处于"瞬态期"（仍在从初始扩散，尚未形成稳定结构）→ 结构性扣分

### 结构稳定性硬约束（非建议，必须遵守）
- 如果种群方差 > 100（高波动），Structural Stability **必须** ≤ 4 分
- 如果种群趋势为 falling（崩溃），Structural Stability **必须** ≤ 3 分

### 物种平衡硬约束（非建议，必须遵守）
- 如果某个物种占比 > 80%，Species Coexistence **必须** ≤ 3 分
- 如果某个物种 < 5%（功能性灭绝），Species Coexistence **必须** ≤ 4 分
- 如果物种均衡度 < 0.3，Species Coexistence **必须** ≤ 4 分
- 最有趣的涌现往往发生在物种势均力敌时
- 如果某物种大量处于休眠孢子状态（统计中标注），说明该物种的生态位可能不适合当前环境

${noveltyReference}

## 输出格式
请严格按照以下 JSON 格式输出，不要输出其他内容：

\`\`\`json
{
  "scores": {
    "species_coexistence": <1-10>,
    "spatial_organization": <1-10>,
    "nutrient_network": <1-10>,
    "structural_stability": <1-10>,
    "dynamic_complexity": <1-10>,
    "aesthetic_emergence": <1-10>
  },
  "composite_score": <加权平均，权重为 0.2, 0.2, 0.15, 0.15, 0.15, 0.15>,
  "emergence_level": "<none|weak|moderate|strong|extraordinary>",
  "description": "<一句话描述你看到的生态现象，不超过60字>",
  "observations": ["<观察1>", "<观察2>"],
  "species_assessment": {
    "explorer": "<该物种的状态描述>",
    "grazer": "<该物种的状态描述>",
    "harvester": "<该物种的状态描述>"
  },
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

请基于截图和数据中的实际证据打分。不确定时偏向保守。`;
}

/**
 * Call the MiMo API with the composite image and evaluation prompt.
 */
export async function evaluateScreenshot(
  compositeScreenshot, params, stats, previousRuns, apiConfig
) {
  const prompt = buildEvalPrompt(params, stats, previousRuns);

  const imageUrl = compositeScreenshot.startsWith('data:')
    ? compositeScreenshot
    : `data:image/png;base64,${compositeScreenshot}`;

  const baseUrl = (apiConfig.baseUrl || 'https://token-plan-sgp.xiaomimimo.com/v1').replace(/\/+$/, '');
  const model = apiConfig.model || 'mimo-v2.5';

  const startTime = Date.now();
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiConfig.apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            { type: 'image_url', image_url: { url: imageUrl } },
          ],
        },
      ],
      max_tokens: 2500,
      temperature: 0.2,
    }),
  });
  const latencyMs = Date.now() - startTime;

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`MiMo API error ${response.status}: ${errText}`);
  }

  const data = await response.json();
  const rawText = data.choices?.[0]?.message?.content || '';
  const tokenUsage = data.usage || null;

  try {
    const parsed = parseLLMResponse(rawText);
    return {
      parsed,
      raw_response: rawText,
      latency_ms: latencyMs,
      token_usage: tokenUsage ? {
        input: tokenUsage.prompt_tokens,
        output: tokenUsage.completion_tokens,
        total: tokenUsage.total_tokens,
      } : null,
    };
  } catch (parseErr) {
    // Re-throw with raw response attached for debugging
    parseErr.raw_response = rawText;
    parseErr.latency_ms = latencyMs;
    parseErr.token_usage = tokenUsage;
    throw parseErr;
  }
}

// ═══════════════════════════════════════
// Prompt helpers
// ═══════════════════════════════════════

function buildHardConstraints(stats) {
  const constraints = [];
  const pops = stats.populations || [];
  const total = stats.totalPopulation || 1;

  // Species balance — enforce via hard caps
  for (let i = 0; i < pops.length; i++) {
    const ratio = pops[i] / total;
    const name = ['Explorer', 'Grazer', 'Harvester'][i];
    if (ratio < 0.05) {
      constraints.push(`⚠️ ${name} 占比仅 ${(ratio * 100).toFixed(1)}%，功能性灭绝。Species Coexistence 必须 ≤ 4 分。`);
    } else if (ratio > 0.8) {
      constraints.push(`⚠️ ${name} 占比 ${(ratio * 100).toFixed(1)}%，完全主导。Species Coexistence 必须 ≤ 3 分。`);
    }
  }

  // Species balance index
  const tv = stats.temporal_variance || {};
  if (tv.species_balance !== undefined && tv.species_balance < 0.3) {
    constraints.push(`⚠️ 物种均衡度 = ${tv.species_balance.toFixed(3)}（严重失衡，1=完全均衡）。Species Coexistence 必须 ≤ 4 分。`);
  }

  // Population variance — high variance means unstable
  if (tv.population_variance !== undefined && tv.population_variance > 100) {
    constraints.push(`⚠️ 种群方差 = ${tv.population_variance.toFixed(1)}（高波动）。Structural Stability 必须 ≤ 4 分。`);
  }

  // Population trend
  if (tv.population_trend === 'falling') {
    constraints.push(`⚠️ 总体种群呈下降趋势，生态系统正在崩溃。Structural Stability 必须 ≤ 3 分。`);
  }

  if (constraints.length === 0) {
    return '## 硬数学指标\n所有指标处于正常范围，无强制约束。';
  }
  return `## 硬数学指标（以下为数学硬约束，不是建议——违反即为错误评分）\n${constraints.join('\n')}`;
}

function buildNoveltyReference(previousRuns) {
  if (previousRuns.length < 3) return '';
  const recent = previousRuns.slice(-5);
  return `## 历史实验描述（用于新颖性判断）
最近 ${recent.length} 轮实验的涌现描述：
${recent.map(r => `- Run ${r.runId} (${r.score}分): ${r.description}`).join('\n')}

如果当前实验的视觉形态与以上所有描述都截然不同，请标记 is_novel=true。`;
}

function formatParams(params) {
  return Object.entries(params)
    .filter(([_, v]) => typeof v === 'number' || typeof v === 'string')
    .map(([k, v]) => `  - ${k}: ${typeof v === 'number' ? v.toFixed?.(4) ?? v : v}`)
    .join('\n');
}

function formatStats(stats) {
  const pops = stats.populations || [];
  const names = ['Explorer', 'Grazer', 'Harvester'];
  const tv = stats.temporal_variance || {};

  const spores = stats.sporeCounts || [];
  const active = stats.activeCounts || [];
  const lines = [
    `  - 总个体数: ${stats.totalPopulation}（含休眠孢子）`,
  ];
  for (let i = 0; i < pops.length; i++) {
    const ratio = stats.totalPopulation > 0 ? (pops[i] / stats.totalPopulation * 100).toFixed(1) : '0.0';
    const sporeStr = spores[i] > 0 ? ` [孢子:${spores[i]}]` : '';
    lines.push(`  - ${names[i]}: ${pops[i]} (${ratio}%), 活跃:${active[i] ?? pops[i]}${sporeStr}, 平均能量: ${stats.avgEnergy?.[i]?.toFixed(2) ?? 'N/A'}`);
  }
  lines.push(`  - 物种均衡度: ${tv.species_balance?.toFixed(3) ?? 'N/A'} (1=完全均衡)`);
  lines.push(`  - 种群趋势: ${tv.population_trend ?? 'N/A'}`);
  lines.push(`  - 种群方差: ${tv.population_variance?.toFixed(1) ?? 'N/A'}`);
  lines.push(`  - 废弃物趋势: ${tv.waste_trend ?? 'N/A'}`);
  if (stats.hotspots) {
    const hsStr = stats.hotspots.map(h => `(${h.x.toFixed(2)},${h.y.toFixed(2)})`).join(' ');
    lines.push(`  - 营养热点位置: ${hsStr}`);
  }
  lines.push(`  - 已运行 tick: ${stats.tickCount}`);

  return lines.join('\n');
}

function buildHistorySummary(previousRuns) {
  if (previousRuns.length === 0) return '';
  const sorted = [...previousRuns].sort((a, b) => b.score - a.score);
  const highs = sorted.slice(0, 3);
  const lows = sorted.slice(-2);

  let text = '## 历史实验参考\n';
  if (highs.length > 0) {
    text += '\n**高分实验：**\n';
    for (const r of highs) {
      text += `- Run ${r.runId}: ${r.score}分 - ${r.description}\n`;
    }
  }
  if (lows.length > 0) {
    text += '\n**低分实验：**\n';
    for (const r of lows) {
      text += `- Run ${r.runId}: ${r.score}分 - ${r.description}\n`;
    }
  }
  return text;
}

// ═══════════════════════════════════════
// Response parsing
// ═══════════════════════════════════════

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
    throw new Error('Failed to parse LLM response as JSON:\n' + text.slice(0, 500));
  }
}
