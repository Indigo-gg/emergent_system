/**
 * Species definitions.
 *
 * Each species has a distinct survival strategy:
 *   - Explorer: fast, wide sensing, burns energy quickly
 *   - Grazer: slow, efficient metabolism, thrives in dense patches
 *   - Harvester: balanced, extra nutrient extraction per consumption
 *
 * Niche differentiation ensures no single species dominates:
 *   - Explorer finds new nutrient sources first
 *   - Grazer survives on scraps the Explorer can't sustain on
 *   - Harvester excels in moderate-density patches
 *
 * Anti-extinction mechanics:
 *   - Minimum population floor (auto-spawn if below threshold)
 *   - Carrying capacity ceiling (reproduction slows near cap)
 *   - Energy reserves prevent instant death from brief starvation
 */
export const SPECIES = [
  {
    id: 0,
    name: 'Explorer',
    color: [30, 100, 100],    // orange-gold (HSB)
    speed: 2.0,
    sensorAngle: 0.5,
    sensorDist: 0.04,
    turnAngle: 0.3,
    metabolismCost: 0.015,     // energy burned per tick
    nutrientGain: 0.4,        // energy gained per nutrient unit consumed
    depositAmount: 0.5,
    maxEnergy: 100,
    reproduceThreshold: 75,
    reproduceCost: 40,
    minPopulation: 30,
    carryingCapacity: 600,
  },
  {
    id: 1,
    name: 'Grazer',
    color: [120, 90, 100],    // green
    speed: 0.8,
    sensorAngle: 0.25,
    sensorDist: 0.025,
    turnAngle: 0.2,
    metabolismCost: 0.004,    // very low burn rate
    nutrientGain: 0.5,
    depositAmount: 0.7,
    maxEnergy: 120,
    reproduceThreshold: 90,
    reproduceCost: 50,
    minPopulation: 25,
    carryingCapacity: 500,
  },
  {
    id: 2,
    name: 'Harvester',
    color: [200, 85, 100],    // cyan-blue
    speed: 1.3,
    sensorAngle: 0.35,
    sensorDist: 0.035,
    turnAngle: 0.25,
    metabolismCost: 0.008,
    nutrientGain: 0.8,        // extracts more per unit
    depositAmount: 0.6,
    maxEnergy: 110,
    reproduceThreshold: 80,
    reproduceCost: 45,
    minPopulation: 20,
    carryingCapacity: 550,
  },
];
