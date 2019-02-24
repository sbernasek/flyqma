from time import time
from os.path import curdir
from clones.validation.arguments import RunArguments
from clones.validation.batch import BatchBenchmark


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = RunArguments(description='Batch benchmark arguments.')
job_path = args['path']
train = args['train_globally']

# ============================= RUN SCRIPT ====================================

start_time = time()


# run each simulation in job file
with open(job_path, 'r') as job_file:

    # run each simulation
    for path in job_file.readlines():

        path = path.strip()

        # load benchmark
        benchmark = BatchBenchmark.load(path)
        benchmark.batch.root = curdir

        # evaluate benchmark
        benchmark.run(train=train)

        # save benchmark
        benchmark.save(path)

# print runtime to standard out
runtime = time() - start_time
print('BATCH COMPLETED IN {:0.2f}.\n'.format(runtime))
