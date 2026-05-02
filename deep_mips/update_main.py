with open('main.py', 'r') as f:
    code = f.read()

code = code.replace(
    'parser.add_argument("--quantize"',
    'parser.add_argument("--fpu", action="store_true", help="Compile using FPU floats")\n    parser.add_argument("--quantize"'
)
code = code.replace(
    'memory_plan = mem_planner.plan(graph, is_quantized=graph.is_quantized)',
    'memory_plan = mem_planner.plan(graph, is_quantized=graph.is_quantized, use_fpu=args.fpu)'
)

with open('main.py', 'w') as f:
    f.write(code)
