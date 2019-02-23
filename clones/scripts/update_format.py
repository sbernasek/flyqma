from clones.validation.arguments import SweepArguments
from clones.validation.sweep import SweepBenchmark

from os.path import join
from glob import glob
from shutil import move


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = SweepArguments(description='SweepBenchmark arguments.')

benchmarking = SweepBenchmark.load(args['path'])


# ============================= EDIT JOB FILES ================================

jobs_path = join(benchmarking.path, 'jobs')
for filepath in glob(join(jobs_path, '[0-9]*.txt')):

    # read existing file contents
    with open(filepath, 'r') as file:
        infile = file.readlines()

    # remove [0-9]*.pkl
    new_lines = [line.strip().rsplit('/', maxsplit=1)[0] for line in infile]

    # write new file contents
    with open(filepath, 'w') as file:
        file.writelines(new_lines)


# ========================= RENAME BATCH FILES ================================

batches_path = join(benchmarking.path, 'batches')
for filepath in glob(join(batches_path, '[0-9]*')):
    src = join(filepath, '{}.pkl'.format(filepath.rsplit('/', maxsplit=1)[-1]))
    dst = join(filepath, 'batch.pkl')
    move(src, dst)





