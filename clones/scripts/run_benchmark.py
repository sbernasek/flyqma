from time import time
from clones.validation.arguments import RunArguments
from clones.validation.batch import BatchBenchmark


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = RunArguments(description='Batch benchmark arguments.')
path = args['path']

# ============================= RUN SCRIPT ====================================

start_time = time()

# load benchmark
benchmark = BatchBenchmark.load(path)

# evaluate benchmark
benchmark.run()

# save benchmark
benchmark.save(path)

# print runtime to standard out
runtime = time() - start_time
print('\BENCHMARK COMPLETED IN {:0.2f}.\n'.format(runtime))
