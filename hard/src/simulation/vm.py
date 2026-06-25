"""
Stack-based virtual machine for executing GEP bytecode on GPU.

The VM is compiled once as a Taichi kernel. Formula changes only modify
the bytecode data array — zero recompilation overhead.

Two VM instances:
1. vm_execute: potential energy U → force via symbolic diff
   Terminal set: dist, density, speed, angle, state[0..3], neighbor_count,
                 avg_nutrient, avg_waste (symmetric quantities)
2. vm_execute_chemotaxis: environment gradient → chemotaxis force F_env
   Terminal set: grad_nut_x/y, grad_waste_x/y, nutrient, waste, speed

Bytecode instruction set:
  OP_CONST idx    — push constants[idx]
  OP_VAR idx      — push vars[idx]
  OP_ADD          — pop b, pop a, push a+b
  OP_SUB          — pop b, pop a, push a-b
  OP_MUL          — pop b, pop a, push a*b
  OP_DIV          — pop b, pop a, push a/b (safe: clamp denominator)
  OP_SIN          — pop a, push sin(a)
  OP_COS          — pop a, push cos(a)
  OP_TANH         — pop a, push tanh(a)
  OP_SQRT         — pop a, push sqrt(|a|)
  OP_ABS          — pop a, push |a|
  OP_NEG          — pop a, push -a
  OP_MAX          — pop b, pop a, push max(a,b)
  OP_MIN          — pop b, pop a, push min(a,b)
  OP_CLAMP        — pop a, push clamp(a, -100, 100)
  OP_HALT         — stop execution
"""

import taichi as ti

# Opcode constants (must match GPU-side)
OP_CONST = 0
OP_VAR = 1
OP_ADD = 10
OP_SUB = 11
OP_MUL = 12
OP_DIV = 13       # safe division: a/b with clamp on b (used in derivatives)
OP_SIN = 20
OP_COS = 21
OP_TANH = 22
OP_SQRT = 23
OP_ABS = 24
OP_NEG = 25
OP_MAX = 30
OP_MIN = 31
OP_CLAMP = 40
OP_HALT = 255

# Variable indices — potential U terminal set
VAR_DIST = 0
VAR_DENSITY = 1
VAR_SPEED = 2
VAR_ANGLE = 3
VAR_STATE_0 = 4
VAR_STATE_1 = 5
VAR_STATE_2 = 6
VAR_STATE_3 = 7
VAR_NEIGHBOR_COUNT = 8
VAR_AVG_NUTRIENT = 9
VAR_AVG_WASTE = 10

# Variable indices — chemotaxis F_env terminal set
CHEMO_VAR_GRAD_NUT_X = 0
CHEMO_VAR_GRAD_NUT_Y = 1
CHEMO_VAR_GRAD_WASTE_X = 2
CHEMO_VAR_GRAD_WASTE_Y = 3
CHEMO_VAR_NUTRIENT = 4
CHEMO_VAR_WASTE = 5
CHEMO_VAR_SPEED = 6


@ti.func
def _get_var(var_idx: ti.i32,
             v0: ti.f32, v1: ti.f32, v2: ti.f32, v3: ti.f32,
             v4: ti.f32, v5: ti.f32, v6: ti.f32, v7: ti.f32,
             v8: ti.f32, v9: ti.f32, v10: ti.f32) -> ti.f32:
    """Select potential variable by index (no dynamic indexing)."""
    result = 0.0
    if var_idx == 0:
        result = v0
    elif var_idx == 1:
        result = v1
    elif var_idx == 2:
        result = v2
    elif var_idx == 3:
        result = v3
    elif var_idx == 4:
        result = v4
    elif var_idx == 5:
        result = v5
    elif var_idx == 6:
        result = v6
    elif var_idx == 7:
        result = v7
    elif var_idx == 8:
        result = v8
    elif var_idx == 9:
        result = v9
    elif var_idx == 10:
        result = v10
    return result


@ti.func
def _get_chemo_var(var_idx: ti.i32,
                   v0: ti.f32, v1: ti.f32, v2: ti.f32, v3: ti.f32,
                   v4: ti.f32, v5: ti.f32, v6: ti.f32) -> ti.f32:
    """Select chemotaxis variable by index."""
    result = 0.0
    if var_idx == 0:
        result = v0
    elif var_idx == 1:
        result = v1
    elif var_idx == 2:
        result = v2
    elif var_idx == 3:
        result = v3
    elif var_idx == 4:
        result = v4
    elif var_idx == 5:
        result = v5
    elif var_idx == 6:
        result = v6
    return result


@ti.func
def _vm_core(bytecode: ti.types.ndarray(),
             constants: ti.types.ndarray(),
             stack_depth: ti.i32) -> ti.f32:
    """Core VM loop — shared by potential and chemotaxis VMs."""
    stack = ti.Vector([0.0] * 16)
    sp = 0
    pc = 0
    halted = 0

    while halted == 0 and pc < 128:
        op = bytecode[pc]

        if op == OP_CONST:
            idx = bytecode[pc + 1]
            sp += 1
            if sp < stack_depth:
                stack[sp] = constants[idx]
            pc += 2

        elif op == OP_ADD:
            if sp >= 2:
                stack[sp - 1] = stack[sp - 1] + stack[sp]
                sp -= 1
            pc += 1

        elif op == OP_SUB:
            if sp >= 2:
                stack[sp - 1] = stack[sp - 1] - stack[sp]
                sp -= 1
            pc += 1

        elif op == OP_MUL:
            if sp >= 2:
                stack[sp - 1] = stack[sp - 1] * stack[sp]
                sp -= 1
            pc += 1

        elif op == OP_DIV:
            if sp >= 2:
                denom = stack[sp]
                if denom > 0:
                    denom = ti.math.max(denom, 1e-7)
                else:
                    denom = ti.math.min(denom, -1e-7)
                stack[sp - 1] = stack[sp - 1] / denom
                sp -= 1
            pc += 1

        elif op == OP_SIN:
            if sp >= 1:
                stack[sp] = ti.sin(stack[sp])
            pc += 1

        elif op == OP_COS:
            if sp >= 1:
                stack[sp] = ti.cos(stack[sp])
            pc += 1

        elif op == OP_TANH:
            if sp >= 1:
                stack[sp] = ti.math.tanh(stack[sp])
            pc += 1

        elif op == OP_SQRT:
            if sp >= 1:
                stack[sp] = ti.sqrt(ti.abs(stack[sp]))
            pc += 1

        elif op == OP_ABS:
            if sp >= 1:
                stack[sp] = ti.abs(stack[sp])
            pc += 1

        elif op == OP_NEG:
            if sp >= 1:
                stack[sp] = -stack[sp]
            pc += 1

        elif op == OP_MAX:
            if sp >= 2:
                stack[sp - 1] = ti.math.max(stack[sp - 1], stack[sp])
                sp -= 1
            pc += 1

        elif op == OP_MIN:
            if sp >= 2:
                stack[sp - 1] = ti.math.min(stack[sp - 1], stack[sp])
                sp -= 1
            pc += 1

        elif op == OP_CLAMP:
            if sp >= 1:
                stack[sp] = ti.math.clamp(stack[sp], -100.0, 100.0)
            pc += 1

        elif op == OP_HALT:
            halted = 1

        else:
            halted = 1

    result = 0.0
    if sp > 0:
        result = stack[sp]
    return result


@ti.func
def vm_execute(bytecode: ti.types.ndarray(),
               constants: ti.types.ndarray(),
               var_dist: ti.f32, var_density: ti.f32,
               var_speed: ti.f32, var_angle: ti.f32,
               var_state0: ti.f32, var_state1: ti.f32,
               var_state2: ti.f32, var_state3: ti.f32,
               var_n_count: ti.f32,
               var_avg_nut: ti.f32, var_avg_wst: ti.f32,
               stack_depth: ti.i32) -> ti.f32:
    """
    Execute potential energy bytecode with OP_VAR resolution.

    Variables are passed as individual floats to avoid ti.Vector
    dynamic indexing issues on GPU.
    """
    # We need to inline the loop here because Taichi doesn't allow
    # passing 11 separate floats through a shared core function easily.
    # Instead, use the _vm_core with a pre-loaded variable vector.
    # Workaround: load variables into a temp array and use _get_var.

    # Actually, Taichi's @ti.func can't easily share the loop with
    # different variable loading. We inline the full loop with _get_var.
    stack = ti.Vector([0.0] * 16)
    sp = 0
    pc = 0
    halted = 0

    while halted == 0 and pc < 128:
        op = bytecode[pc]

        if op == OP_CONST:
            idx = bytecode[pc + 1]
            sp += 1
            if sp < stack_depth:
                stack[sp] = constants[idx]
            pc += 2

        elif op == OP_VAR:
            idx = bytecode[pc + 1]
            sp += 1
            if sp < stack_depth:
                stack[sp] = _get_var(
                    idx,
                    var_dist, var_density, var_speed, var_angle,
                    var_state0, var_state1, var_state2, var_state3,
                    var_n_count, var_avg_nut, var_avg_wst
                )
            pc += 2

        elif op == OP_ADD:
            if sp >= 2:
                stack[sp - 1] = stack[sp - 1] + stack[sp]
                sp -= 1
            pc += 1

        elif op == OP_SUB:
            if sp >= 2:
                stack[sp - 1] = stack[sp - 1] - stack[sp]
                sp -= 1
            pc += 1

        elif op == OP_MUL:
            if sp >= 2:
                stack[sp - 1] = stack[sp - 1] * stack[sp]
                sp -= 1
            pc += 1

        elif op == OP_DIV:
            if sp >= 2:
                denom = stack[sp]
                if denom > 0:
                    denom = ti.math.max(denom, 1e-7)
                else:
                    denom = ti.math.min(denom, -1e-7)
                stack[sp - 1] = stack[sp - 1] / denom
                sp -= 1
            pc += 1

        elif op == OP_SIN:
            if sp >= 1:
                stack[sp] = ti.sin(stack[sp])
            pc += 1

        elif op == OP_COS:
            if sp >= 1:
                stack[sp] = ti.cos(stack[sp])
            pc += 1

        elif op == OP_TANH:
            if sp >= 1:
                stack[sp] = ti.math.tanh(stack[sp])
            pc += 1

        elif op == OP_SQRT:
            if sp >= 1:
                stack[sp] = ti.sqrt(ti.abs(stack[sp]))
            pc += 1

        elif op == OP_ABS:
            if sp >= 1:
                stack[sp] = ti.abs(stack[sp])
            pc += 1

        elif op == OP_NEG:
            if sp >= 1:
                stack[sp] = -stack[sp]
            pc += 1

        elif op == OP_MAX:
            if sp >= 2:
                stack[sp - 1] = ti.math.max(stack[sp - 1], stack[sp])
                sp -= 1
            pc += 1

        elif op == OP_MIN:
            if sp >= 2:
                stack[sp - 1] = ti.math.min(stack[sp - 1], stack[sp])
                sp -= 1
            pc += 1

        elif op == OP_CLAMP:
            if sp >= 1:
                stack[sp] = ti.math.clamp(stack[sp], -100.0, 100.0)
            pc += 1

        elif op == OP_HALT:
            halted = 1

        else:
            halted = 1

    result = 0.0
    if sp > 0:
        result = stack[sp]
    return result


@ti.func
def vm_execute_chemotaxis(bytecode: ti.types.ndarray(),
                          constants: ti.types.ndarray(),
                          var_grad_nx: ti.f32, var_grad_ny: ti.f32,
                          var_grad_wx: ti.f32, var_grad_wy: ti.f32,
                          var_nutrient: ti.f32, var_waste: ti.f32,
                          var_speed: ti.f32,
                          stack_depth: ti.i32) -> ti.f32:
    """
    Execute chemotaxis bytecode — environment gradient → force.

    Terminal set: grad_nut_x, grad_nut_y, grad_waste_x, grad_waste_y,
                  nutrient, waste, speed
    """
    stack = ti.Vector([0.0] * 16)
    sp = 0
    pc = 0
    halted = 0

    while halted == 0 and pc < 128:
        op = bytecode[pc]

        if op == OP_CONST:
            idx = bytecode[pc + 1]
            sp += 1
            if sp < stack_depth:
                stack[sp] = constants[idx]
            pc += 2

        elif op == OP_VAR:
            idx = bytecode[pc + 1]
            sp += 1
            if sp < stack_depth:
                stack[sp] = _get_chemo_var(
                    idx,
                    var_grad_nx, var_grad_ny, var_grad_wx, var_grad_wy,
                    var_nutrient, var_waste, var_speed
                )
            pc += 2

        elif op == OP_ADD:
            if sp >= 2:
                stack[sp - 1] = stack[sp - 1] + stack[sp]
                sp -= 1
            pc += 1

        elif op == OP_SUB:
            if sp >= 2:
                stack[sp - 1] = stack[sp - 1] - stack[sp]
                sp -= 1
            pc += 1

        elif op == OP_MUL:
            if sp >= 2:
                stack[sp - 1] = stack[sp - 1] * stack[sp]
                sp -= 1
            pc += 1

        elif op == OP_DIV:
            if sp >= 2:
                denom = stack[sp]
                if denom > 0:
                    denom = ti.math.max(denom, 1e-7)
                else:
                    denom = ti.math.min(denom, -1e-7)
                stack[sp - 1] = stack[sp - 1] / denom
                sp -= 1
            pc += 1

        elif op == OP_SIN:
            if sp >= 1:
                stack[sp] = ti.sin(stack[sp])
            pc += 1

        elif op == OP_COS:
            if sp >= 1:
                stack[sp] = ti.cos(stack[sp])
            pc += 1

        elif op == OP_TANH:
            if sp >= 1:
                stack[sp] = ti.math.tanh(stack[sp])
            pc += 1

        elif op == OP_SQRT:
            if sp >= 1:
                stack[sp] = ti.sqrt(ti.abs(stack[sp]))
            pc += 1

        elif op == OP_ABS:
            if sp >= 1:
                stack[sp] = ti.abs(stack[sp])
            pc += 1

        elif op == OP_NEG:
            if sp >= 1:
                stack[sp] = -stack[sp]
            pc += 1

        elif op == OP_MAX:
            if sp >= 2:
                stack[sp - 1] = ti.math.max(stack[sp - 1], stack[sp])
                sp -= 1
            pc += 1

        elif op == OP_MIN:
            if sp >= 2:
                stack[sp - 1] = ti.math.min(stack[sp - 1], stack[sp])
                sp -= 1
            pc += 1

        elif op == OP_CLAMP:
            if sp >= 1:
                stack[sp] = ti.math.clamp(stack[sp], -100.0, 100.0)
            pc += 1

        elif op == OP_HALT:
            halted = 1

        else:
            halted = 1

    result = 0.0
    if sp > 0:
        result = stack[sp]
    return result
