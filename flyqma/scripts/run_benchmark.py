from time import time
from os.path import curdir
from clones.validation.arguments import RunArguments
from clones.validation.batch import BatchBenchmark


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = RunArguments(description='Batch benchmark arguments.')
path = args['path']
data = args['save_data']

# ============================= RUN SCRIPT ====================================

start_time = time()

# load benchmark
benchmark = BatchBenchmark.load(path)
benchmark.batch.root = curdir

# evaluate benchmark
benchmark.run()

# save benchmark
benchmark.save(path, data=data)

# print runtime to standard out
runtime = time() - start_time
print('BENCHMARK COMPLETED IN {:0.2f}.\n'.format(runtime))
