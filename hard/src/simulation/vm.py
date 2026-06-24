"""
Stack-based virtual machine for executing GEP bytecode on GPU.

The VM is compiled once as a Taichi kernel. Formula changes only modify
the bytecode data array — zero recompilation overhead.

Bytecode instruction set:
  OP_CONST idx    — push constants[idx]
  OP_VAR idx      — push vars[idx]
  OP_ADD          — pop b, pop a, push a+b
  OP_SUB          — pop b, pop a, push a-b
  OP_MUL          — pop b, pop a, push a*b
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

# Variable indices (must match terminal set order)
VAR_DIST = 0
VAR_DENSITY = 1
VAR_SPEED = 2
VAR_ANGLE = 3
VAR_STATE_0 = 4
VAR_STATE_1 = 5
VAR_STATE_2 = 6
VAR_STATE_3 = 7
VAR_NEIGHBOR_COUNT = 8


@ti.func
def vm_execute(bytecode: ti.types.ndarray(),
               constants: ti.types.ndarray(),
               vars: ti.types.ndarray(),
               stack_depth: ti.i32) -> ti.f32:
    """
    Execute bytecode in a stack-based VM on GPU.

    Args:
        bytecode: int array of opcodes + operands
        constants: float array of constant values
        vars: float array of variable values (dist, density, etc.)
        stack_depth: max stack depth (for bounds checking)

    Returns:
        Final value on top of stack
    """
    # Fixed-size stack — must be small for register allocation
    stack = ti.Vector([0.0] * 16)
    sp = 0  # stack pointer
    pc = 0  # program counter

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
                stack[sp] = vars[idx]
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
            # Unknown opcode — treat as HALT
            halted = 1

    # Return top of stack
    result = 0.0
    if sp > 0:
        result = stack[sp]
    return result
