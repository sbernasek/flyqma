from os.path import join, abspath, relpath, isdir
from os import mkdir, chmod, pardir
import shutil

import numpy as np
import dill as pickle

from growth.sweep.sweep import Sweep
from .batch import BatchBenchmark
from .io import Pickler


class SweepBenchmark(Pickler):

    """
    Attributes:

        sweep_path (str) - path to parent growth.sweep.Sweep directory

        path (str) - path to benchmark subdirectory

        batches (2D np.ndarray[growth.Batch]) - batches of growth replicates

        scales (np.ndarray[float]) - fluorescence scales

        num_replicates (int) - number of fluorescence replicates

        script_name (str) - name of run script

    """

    def __init__(self, sweep_path,
                 num_scales=9,
                 num_replicates=10,
                 script_name='run_benchmark.py'):

        # load and set growth sweep
        self.sweep_path = sweep_path
        self.batches = Sweep.load(sweep_path).batches

        # set fluorescence scales
        self.scales = np.linspace(2, 10, num_scales)

        # set number of fluorescence replicates
        self.num_replicates = num_replicates

        # instantiate dictionary of paths to each BatchBenchmark object
        self.benchmark_paths = {}

        # set script name
        self.script_name = script_name

    @staticmethod
    def load(sweep_path):
        """
        Load job.

        Args:

            sweep_path (str) - path to growth.Sweep directory

        """
        path = join(sweep_path, 'benchmark')
        with open(join(path, 'benchmark_job.pkl'), 'rb') as file:
            job = pickle.load(file)
        job.sweep_path = sweep_path
        job.path = path
        return job

    @property
    def num_scales(self):
        """ Number of fluorescence scales. """
        return self.scales.size

    @staticmethod
    def build_run_script(path, script_name):
        """
        Writes bash run script for local use.

        Args:

            path (str) - path to benchmarking top directory

            script_name (str) - name of run script

        """

        # define paths
        path = abspath(path)
        run_path = join(path, '..')
        job_script_path = join(path, 'scripts', 'run.sh')

        # copy run script to scripts directory
        scripts_path = abspath(__file__).rsplit('/', maxsplit=2)[0]
        run_script = join(scripts_path, 'scripts', script_name)
        shutil.copy(run_script, join(path, 'scripts'))

        # declare outer script that reads PATH from file
        job_script = open(job_script_path, 'w')
        job_script.write('#!/bin/bash\n')

        # move to job directory
        job_script.write('cd {:s} \n\n'.format(run_path))

        # run each batch
        job_script.write('echo "Starting all batches at `date`"\n')
        job_script.write('while read P; do\n')
        job_script.write('echo "Processing batch ${P}"\n')
        job_script.write('python ./scripts/{:s}'.format(script_name)+' ${P} ')
        job_script.write('done < ./jobs/index.txt \n')
        job_script.write('echo "Job completed at `date`"\n')
        job_script.write('exit\n')

        # close the file
        job_script.close()

        # change the permissions
        chmod(job_script_path, 0o755)

    @staticmethod
    def build_submission_script(path,
                                script_name,
                                walltime=10,
                                allocation='p30653',
                                cores=1,
                                memory=4):
        """
        Writes job submission script for QUEST.

        Args:

            path (str) - path to benchmarking top directory

            script_name (str) - name of run script

            save_history (bool) - if True, save simulation history

            walltime (int) - estimated job run time

            allocation (str) - project allocation, e.g. p30653 (comp. bio)

            cores (int) - number of cores per batch

            memory (int) - memory per batch, GB

        """

        # define paths
        path = abspath(path)
        run_path = join(path, '..')
        job_script_path = join(path, 'scripts', 'submit.sh')

        # copy run script to scripts directory
        scripts_path = abspath(__file__).rsplit('/', maxsplit=2)[0]
        run_script = join(scripts_path, 'scripts', script_name)
        shutil.copy(run_script, join(path, 'scripts'))

        # determine queue
        if walltime <= 4:
            queue = 'short'
        elif walltime <= 48:
            queue = 'normal'
        else:
            queue = 'long'

        # declare outer script that reads PATH from file
        job_script = open(job_script_path, 'w')
        job_script.write('#!/bin/bash\n')

        # move to benchmarking directory
        job_script.write('cd {:s} \n\n'.format(path))

        # begin outer script for processing job
        job_script.write('while IFS=$\'\\t\' read P\n')
        job_script.write('do\n')
        job_script.write('b_id=$(echo $(basename ${P}) | cut -f 1 -d \'.\')\n')
        job_script.write('   JOB=`msub - << EOJ\n\n')

        # =========== begin submission script for individual job ==============
        job_script.write('#! /bin/bash\n')
        job_script.write('#MSUB -A {:s} \n'.format(allocation))
        job_script.write('#MSUB -q {:s} \n'.format(queue))
        job_script.write('#MSUB -l walltime={0:02d}:00:00 \n'.format(walltime))
        job_script.write('#MSUB -m abe \n')
        #job_script.write('#MSUB -M sebastian@u.northwestern.edu \n')
        job_script.write('#MSUB -o ./log/${b_id}/outlog \n')
        job_script.write('#MSUB -e ./log/${b_id}/errlog \n')
        job_script.write('#MSUB -N ${b_id} \n')
        job_script.write('#MSUB -l nodes=1:ppn={:d} \n'.format(cores))
        job_script.write('#MSUB -l mem={:d}gb \n\n'.format(memory))

        # load python module and clones virtual environment
        job_script.write('module load python/anaconda3.6\n')
        job_script.write('source activate ~/pythonenvs/clones_env\n\n')

        # move to job directory
        job_script.write('cd {:s} \n\n'.format(run_path))

        # run script
        job_script.write('python ./scripts/{:s}'.format(script_name)+' ${P} \n')
        job_script.write('EOJ\n')
        job_script.write('`\n\n')
        # ============= end submission script for individual job --============

        # print job id
        job_script.write('done < ./jobs/index.txt \n')
        job_script.write('echo "All jobs submitted as of `date`"\n')
        job_script.write('exit\n')

        # close the file
        job_script.close()

        # change the permissions
        chmod(job_script_path, 0o755)

    def build_batches(self):
        """ Writes benchmark paths for each batch. """

        # get directories for all batches and logs
        batches_dir = join(self.path, 'batches')
        jobs_dir = join(self.path, 'jobs')
        logs_dir = join(self.path, 'log')

        # create index file for batches
        index_path = join(jobs_dir, 'index.txt')
        index = open(index_path, 'w')

        # write file containing benchmark paths for each batch
        for batch_id, batch in enumerate(self.batches.ravel()):

            # append job file to index
            job_path = join(jobs_dir, '{:d}.txt'.format(batch_id))
            index.write('{:s}\n'.format(relpath(job_path, self.sweep_path)))

            # open job file
            job_file = open(job_path, 'w')

            # write batch benchmark paths to job file
            for scale_id in range(self.num_scales):
                benchmark_path = self.benchmark_paths[batch_id][scale_id]
                job_file.write('{:s}\n'.format(benchmark_path))

            # create log directory for job
            mkdir(join(logs_dir, '{:d}'.format(batch_id)))

            # close job file
            job_file.close()
            chmod(job_path, 0o755)

        # close index file
        index.close()
        chmod(index_path, 0o755)

    def make_subdirectory(self):
        """ Create job subdirectory. """

        # create directory (overwrite existing one)
        path = join(self.sweep_path, 'benchmark')
        if isdir(path):
            shutil.rmtree(path)
        mkdir(path)
        self.path = path

        # make subdirectories for simulations and scripts
        mkdir(join(path, 'scripts'))
        mkdir(join(path, 'batches'))
        mkdir(join(path, 'jobs'))
        mkdir(join(path, 'log'))

    def build(self,
              walltime=10,
              allocation='p30653',
              cores=1,
              memory=4,
              **kwargs):
        """
        Build job directory tree. Instantiates and saves a simulation instance for each parameter set, then generates a single shell script to submit each simulation as a separate job.

        Args:

            walltime (int) - estimated job run time

            allocation (str) - project allocation

            cores (int) - number of cores per batch

            memory (int) - memory per batch, GB

            kwargs (dict) - keyword arguments for benchmarking

        """

        # create benchmarking subdirectory
        self.make_subdirectory()

        # store parameters (e.g. pulse conditions)
        self.kwargs = kwargs

        # build batch benchmarks
        for batch_id, batch in enumerate(self.batches.ravel()):

            # make batch directory
            batch_path = join(self.path, 'batches', '{:d}'.format(batch_id))
            mkdir(batch_path)
            benchmark_paths = {}

            # make benchmarks for each scale of batch
            for scale_id, scale in enumerate(self.scales):

                # store benchmark path
                benchmark_path = join(batch_path, '{:d}.pkl'.format(scale_id))
                benchmark_paths[scale_id] = relpath(benchmark_path, self.sweep_path)

                # build benchmark for current scale
                benchmark_arg = (benchmark_path, batch, scale, self.num_replicates)
                self.build_benchmark(*benchmark_arg, **kwargs)

            # store scale paths
            self.benchmark_paths[batch_id] = benchmark_paths

        # save serialized job
        #with open(join(self.path, 'benchmark_job.pkl'), 'wb') as file:
        #    pickle.dump(self, file, protocol=-1)
        self.save(join(self.path, 'benchmark_job.pkl'))

        # build parameter file for each batch
        self.build_batches()

        # build job run script
        self.build_run_script(self.path, self.script_name)

        # build job submission script
        self.build_submission_script(self.path,
                                     self.script_name,
                                     walltime=walltime,
                                     allocation=allocation,
                                     cores=cores,
                                     memory=memory)

    @classmethod
    def build_benchmark(cls, path, batch, scale, num_replicates, **kwargs):
        """
        Builds and saves a BatchBenchmark instance for a given batch and scale.

        Args:

            path (str) - path to benchmark object

            batch (growth.sweep.Batch) - batch of growth replicates

            scale (float) - fluorescence scale

            num_replicates (int) - number of fluorescence replicates

            kwargs: keyword arguments for BatchBenchmark

        """

        # instantiate benchmark
        benchmark = BatchBenchmark(batch, scale, num_replicates, **kwargs)

        # save benchmark
        benchmark.save(path)

    def load_benchmark(self, batch_id, scale_id):
        """
        Load simulation instance from file.

        Args:

            batch_id (int) - batch index

            scale_id (int) - scale index

        Returns:

            benchmark (BatchBenchmark)

        """
        path = join(self.path, self.benchmark_paths[batch_id][scale_id])
        benchmark = BatchBenchmark.load(path)
        return benchmark
