from clones.validation.arguments import SweepArguments
from clones.validation.sweep import SweepBenchmark


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = SweepArguments(description='Benchmark sweep options.')
sweep_path = args['path']
min_scale = args['min_scale']
max_scale = args['max_scale']
num_scales = args['num_scales']
num_replicates = args['num_replicates']

# ============================= RUN SCRIPT ====================================

# instantiate SweepBenchmark object
sweep = SweepBenchmark(
    sweep_path,
    min_scale=min_scale,
    max_scale=max_scale,
    num_scales=num_scales,
    num_replicates=num_replicates)

# build SweepBenchmark
sweep.build(
    rule=args['rule'],
    twolevel=args['twolevel'],
    walltime=args['walltime'],
    cores=args['cores'],
    memory=args['memory'],
    allocation=args['allocation'])
