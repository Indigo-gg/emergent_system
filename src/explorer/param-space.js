/**
 * Parameter space definition for the Ecosystem Explorer v2.
 *
 * 13 parameters across 4 categories:
 *   - Trail system (2)
 *   - Nutrient system (5)
 *   - Waste/toxin system (4)
 *   - Population (1)
 *   - Hotspot drift (1)
 */
export const PARAM_SPACE = {
  // ─── Trail system ───
  trailDiffuseRate: {
    min: 0.05, max: 0.5, step: 0.05,
    description: '信息素扩散速率',
    category: 'trail',
  },
  trailDecayRate: {
    min: 0.90, max: 0.999, step: 0.005,
    description: '信息素衰减率',
    category: 'trail',
  },

  // ─── Nutrient system ───
  nutrientDiffuseRate: {
    min: 0.02, max: 0.4, step: 0.02,
    description: '营养扩散速率',
    category: 'nutrient',
  },
  nutrientDecayRate: {
    min: 0.98, max: 0.9999, step: 0.002,
    description: '营养衰减率',
    category: 'nutrient',
  },
  nutrientInjectInterval: {
    min: 20, max: 300, step: 10,
    description: '营养注入间隔（tick）',
    category: 'nutrient',
  },
  nutrientPatchCount: {
    min: 1, max: 15, step: 1,
    description: '每次注入斑块数',
    category: 'nutrient',
  },
  nutrientPatchAmount: {
    min: 0.3, max: 5.0, step: 0.2,
    description: '斑块中心浓度',
    category: 'nutrient',
  },

  // ─── Waste/toxin system ───
  wasteProductionRate: {
    min: 0.02, max: 0.5, step: 0.02,
    description: '废弃物产生率（每单位营养消耗产生的废弃物）',
    category: 'waste',
  },
  wasteDecayRate: {
    min: 0.98, max: 0.999, step: 0.002,
    description: '废弃物衰减率。越高→毒素越持久',
    category: 'waste',
  },
  wasteMetabolismFactor: {
    min: 0.5, max: 5.0, step: 0.25,
    description: '废弃物对代谢的放大系数。高→在毒素区代谢暴增',
    category: 'waste',
  },
  wasteRepelStrength: {
    min: 0.1, max: 2.0, step: 0.1,
    description: '废弃物的排斥强度。高→生物更积极躲避毒素',
    category: 'waste',
  },

  // ─── Hotspot drift ───
  nutrientDriftSpeed: {
    min: 0, max: 0.005, step: 0.0005,
    description: '营养热点基础漂移速度。实际速度会周期性变化（0.3x~1.7x）',
    category: 'environment',
  },

  // ─── Terrain ───
  terrainScale: {
    min: 0.01, max: 0.1, step: 0.005,
    description: '地形噪声缩放率。小→大块地形，小→碎片化地形',
    category: 'terrain',
  },
  terrainWallThreshold: {
    min: 0.55, max: 0.85, step: 0.025,
    description: '墙壁阈值。越低→墙壁越多',
    category: 'terrain',
  },

  // ─── Population ───
  initialPerSpecies: {
    min: 30, max: 200, step: 10,
    description: '每物种初始个体数',
    category: 'population',
  },
};

/**
 * Default parameter values.
 */
export const DEFAULTS = {
  trailDiffuseRate: 0.2,
  trailDecayRate: 0.97,
  nutrientDiffuseRate: 0.08,
  nutrientDecayRate: 0.999,
  nutrientInjectInterval: 60,
  nutrientPatchCount: 5,
  nutrientPatchAmount: 1.5,
  wasteProductionRate: 0.5,
  wasteDecayRate: 0.998,
  wasteMetabolismFactor: 2.5,
  wasteRepelStrength: 1.2,
  nutrientDriftSpeed: 0.0008,
  terrainScale: 0.04,
  terrainWallThreshold: 0.7,
  initialPerSpecies: 80,
};
