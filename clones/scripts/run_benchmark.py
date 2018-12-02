from time import time
from os.path import curdir
from clones.validation.arguments import RunArguments
from clones.validation.batch import BatchBenchmark


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = RunArguments(description='Batch benchmark arguments.')
path = args['path']
save_measurements = args['save_measurements']

# ============================= RUN SCRIPT ====================================

start_time = time()

# load benchmark
benchmark = BatchBenchmark.load(path)
benchmark.batch.root = curdir

# evaluate benchmark
benchmark.run()

# save benchmark
benchmark.save(path, save_measurements=save_measurements)

# print runtime to standard out
runtime = time() - start_time
print('\BENCHMARK COMPLETED IN {:0.2f}.\n'.format(runtime))
