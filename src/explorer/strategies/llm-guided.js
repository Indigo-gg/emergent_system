/**
 * LLM-Guided exploration strategy (Phase B).
 *
 * Uses the LLM to analyze experiment history and suggest next parameters.
 * Includes phase boundary detection: if adjacent runs have very different
 * scores with similar params, the LLM is warned about non-linearity.
 */
import { PARAM_SPACE } from '../param-space.js';

export class LLMGuidedStrategy {
  constructor(paramSpace = PARAM_SPACE, apiConfig = {}) {
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
    const phaseBoundaryWarning = detectPhaseBoundaries(recent);

    return `你是一个复杂生态系统实验科学家。你正在调整一个多物种粘菌生态系统的参数，寻找涌现现象。

## 实验系统
三物种粘菌生态系统（15000 tick/轮）：
- Explorer（橙金）：快速广感知，高代谢
- Grazer（绿色）：慢速低代谢，高效利用
- Harvester（青蓝）：平衡型，高营养提取

系统包含：废弃物/毒素场、移动营养热点、休眠孢子机制、跨物种信息素交互

## 参数空间
${JSON.stringify(this.paramSpace, null, 2)}

## 最近 ${recent.length} 轮实验结果
${recent.map(r =>
  `- Run ${r.runId}: score=${r.score}, level=${r.level}
   params: ${JSON.stringify(r.params)}
   评语: ${r.description}`
).join('\n')}

## 历史最佳结果
${best.map(r =>
  `- Run ${r.runId}: score=${r.score}, ${r.description}
   params: ${JSON.stringify(r.params)}`
).join('\n')}

${phaseBoundaryWarning}

## 你的任务
基于以上实验记录，建议下一轮实验的参数。目标是找到更强的涌现现象。

关键研究问题：
- 废弃物产生率如何影响动态脉冲和扩张收缩循环？
- 营养热点漂移速度如何影响宏观迁移模式？
- 物种间的信息素排斥/吸引如何塑造领地边界？
- 什么样的参数组合能在 15000 tick 内达到"优化期"或"生态期"？

请输出一个 JSON 对象：

\`\`\`json
{
  "reasoning": "<你的分析思路，100字以内>",
  "params": {
    "trailDiffuseRate": <值>,
    "trailDecayRate": <值>,
    "nutrientDiffuseRate": <值>,
    "nutrientDecayRate": <值>,
    "nutrientInjectInterval": <值>,
    "nutrientPatchCount": <值>,
    "nutrientPatchRadius": <值>,
    "nutrientPatchAmount": <值>,
    "initialPerSpecies": <值>
  },
  "hypothesis": "<你期望看到什么生态现象，50字以内>"
}
\`\`\`

[CRITICAL] 非线性系统警告：
1. 生态系统存在"相变悬崖"——参数微小变化可能导致物种灭绝或生态崩溃
2. 如果两组相近参数的得分差异 > 3 分，你可能找到了相变边界
3. 发现相变边界时：围绕边界做极小步长采样（step 的 1/4），不要向外扩展
4. 所有参数必须在定义的 min/max 范围内
5. 不要重复已试过的参数组合`;
  }

  async callLLM(prompt) {
    const baseUrl = (this.apiConfig.baseUrl || 'https://token-plan-sgp.xiaomimimo.com/v1').replace(/\/+$/, '');
    const model = this.apiConfig.model || 'mimo-v2.5';

    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiConfig.apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 1000,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`LLM guidance API error ${response.status}: ${err}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || '';
  }

  parseParamSuggestion(text) {
    const jsonBlockMatch = text.match(/```json\s*([\s\S]*?)```/);
    if (jsonBlockMatch) return JSON.parse(jsonBlockMatch[1].trim());
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start !== -1 && end !== -1) return JSON.parse(text.slice(start, end + 1));
    throw new Error('Failed to parse LLM parameter suggestion:\n' + text.slice(0, 500));
  }
}

/**
 * Detect possible phase boundaries in recent runs.
 */
function detectPhaseBoundaries(runs) {
  if (runs.length < 3) return '';

  const boundaries = [];
  for (let i = 1; i < runs.length; i++) {
    const prev = runs[i - 1];
    const curr = runs[i];
    const scoreDiff = Math.abs(curr.score - prev.score);

    if (scoreDiff > 3) {
      const paramDist = computeParamDistance(prev.params, curr.params);
      if (paramDist < 0.3) {
        boundaries.push({
          run1: prev.runId, run2: curr.runId, scoreDiff,
          params1: prev.params, params2: curr.params,
        });
      }
    }
  }

  if (boundaries.length === 0) return '';

  return `## ⚠️ 检测到可能的相变边界
以下实验对的参数相近但得分差异巨大：
${boundaries.map(b =>
  `- Run ${b.run1}→${b.run2} (${b.scoreDiff.toFixed(1)}分差异): ` +
  `nutrientPatch ${b.params1.nutrientPatchCount}→${b.params2.nutrientPatchCount}, ` +
  `trailDecay ${b.params1.trailDecayRate}→${b.params2.trailDecayRate}`
).join('\n')}

建议：在这些参数值之间做极小步长的精细采样。`;
}

function computeParamDistance(p1, p2) {
  const keys = ['trailDecayRate', 'nutrientPatchCount', 'nutrientPatchAmount'];
  let sum = 0;
  for (const k of keys) {
    const v1 = p1[k] ?? 0;
    const v2 = p2[k] ?? 0;
    const diff = (v1 - v2) / (Math.abs(v1) + Math.abs(v2) + 0.001);
    sum += diff * diff;
  }
  return Math.sqrt(sum / keys.length);
}
