from clones.validation.arguments import SweepArguments
from clones.validation.sweep import SweepBenchmark


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = SweepArguments(description='Benchmark sweep options.')
sweep_path = args['path']
min_ambiguity = args['min_ambiguity']
max_ambiguity = args['max_ambiguity']
num_ambiguities = args['num_ambiguities']
num_replicates = args['num_replicates']

# ============================= RUN SCRIPT ====================================

# instantiate SweepBenchmark object
sweep = SweepBenchmark(
    sweep_path,
    min_ambiguity=min_ambiguity,
    max_ambiguity=max_ambiguity,
    num_ambiguities=num_ambiguities,
    num_replicates=num_replicates)

# build SweepBenchmark
sweep.build(
    logratio=args['logratio'],
    train_globally=args['train_globally'],
    walltime=args['walltime'],
    cores=args['cores'],
    memory=args['memory'],
    allocation=args['allocation'],
    attribute='clonal_marker')
