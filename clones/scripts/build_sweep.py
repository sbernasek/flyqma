from clones.validation.arguments import SweepArguments
from clones.validation.sweep import SweepBenchmark


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = SweepArguments(description='Benchmark sweep options.')
sweep_path = args['path']
num_scales = args['num_scales']
num_replicates = args['num_replicates']

# ============================= RUN SCRIPT ====================================

# instantiate SweepBenchmark object
sweep = SweepBenchmark(
    sweep_path,
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
