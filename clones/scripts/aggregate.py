from time import time
from clones.validation.arguments import SweepArguments
from clones.validation.sweep import SweepBenchmark


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = SweepArguments(description='SweepBenchmark arguments.')

# ============================= RUN SCRIPT ====================================

start = time()
sweep_benchmark = SweepBenchmark.load(args['path'])
sweep_benchmark.aggregate()
runtime = time() - start

print('AGGREGATION COMPLETED IN {:0.2f} seconds.'.format(runtime))
