from os.path import join
from dill import pickle
from .batch import BatchBenchmark, Pickler


class SweepBenchmark(Pickler):

    def __init__(self, sweep):
        self.sweep = sweep

    @staticmethod
    def load(path):
        """ Load job from target <path>. """
        with open(join(path, 'benchmark.pkl'), 'rb') as file:
            job = pickle.load(file)
        job.sweep.path = path
        return job



    @staticmethod
    def build_run_script(path, script_name, save_history):
        """
        Writes bash run script for local use.

        Args:

            path (str) - path to simulation top directory

            script_name (str) - name of run script

            save_history (bool) - if True, save simulation history

        """

        # define paths
        path = abspath(path)
        job_script_path = join(path, 'scripts', 'run.sh')

        # copy run script to scripts directory
        run_script = abspath(__file__).rsplit('/', maxsplit=2)[0]
        run_script = join(run_script, 'scripts', script_name)
        shutil.copy(run_script, join(path, 'scripts'))

        # declare outer script that reads PATH from file
        job_script = open(job_script_path, 'w')
        job_script.write('#!/bin/bash\n')

        # move to job directory
        job_script.write('cd {:s} \n\n'.format(path))

        # run each batch
        job_script.write('echo "Starting all batches at `date`"\n')
        job_script.write('while read P; do\n')
        job_script.write('echo "Processing batch ${P}"\n')
        job_script.write('python ./scripts/{:s}'.format(script_name)+' ${P} ')
        args = (save_history,)
        job_script.write('-s {:d}\n'.format(*args))
        job_script.write('done < ./batches/index.txt \n')
        job_script.write('echo "Job completed at `date`"\n')
        job_script.write('exit\n')

        # close the file
        job_script.close()

        # change the permissions
        chmod(job_script_path, 0o755)

    @staticmethod
    def build_submission_script(path,
                                script_name,
                                save_history=True,
                                walltime=10,
                                allocation='p30653',
                                cores=1,
                                memory=4):
        """
        Writes job submission script for QUEST.

        Args:

            path (str) - path to simulation top directory

            script_name (str) - name of run script

            save_history (bool) - if True, save simulation history

            walltime (int) - estimated job run time

            allocation (str) - project allocation, e.g. p30653 (comp. bio)

            cores (int) - number of cores per batch

            memory (int) - memory per batch, GB

        """

        # define paths
        path = abspath(path)
        job_script_path = join(path, 'scripts', 'submit.sh')

        # copy run script to scripts directory
        run_script = abspath(__file__).rsplit('/', maxsplit=2)[0]
        run_script = join(run_script, 'scripts', script_name)
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

        # move to job directory
        job_script.write('cd {:s} \n\n'.format(path))

        # begin outer script for processing job
        job_script.write('while IFS=$\'\\t\' read P\n')
        job_script.write('do\n')
        job_script.write('b_id=$(echo $(basename ${P}) | cut -f 1 -d \'.\')\n')
        job_script.write('   JOB=`msub - << EOJ\n\n')

        # =========== begin submission script for individual batch ============
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

        # load python module and metabolism virtual environment
        job_script.write('module load python/anaconda3.6\n')
        job_script.write('source activate growth\n\n')

        # move to job directory
        job_script.write('cd {:s} \n\n'.format(path))

        # run script
        job_script.write('python ./scripts/{:s}'.format(script_name)+' ${P} ')
        args = (save_history,)
        job_script.write('-s {:d}\n'.format(*args))
        job_script.write('EOJ\n')
        job_script.write('`\n\n')
        # ============= end submission script for individual batch ============

        # print job id
        #job_script.write('echo "JobID = ${JOB} submitted on `date`"\n')
        job_script.write('done < ./batches/index.txt \n')
        job_script.write('echo "All batches submitted as of `date`"\n')
        job_script.write('exit\n')

        # close the file
        job_script.close()

        # change the permissions
        chmod(job_script_path, 0o755)





    def build_batches(self):
        """
        Creates directory and writes simulation paths for each batch.

        Args:

            batch_size (int) - number of simulations per batch

        """

        # get directories for all batches and logs
        batches_dir = join(self.path, 'batches')
        logs_dir = join(self.path, 'log')

        # create index file for batches
        index_path = join(batches_dir, 'index.txt')
        index = open(index_path, 'w')

        # write file containing simulation paths for each batch
        for i, simulation_path in self.simulation_paths.items():

            # determine batch ID
            batch_id = i // self.batch_size

            # process new batch
            if i % self.batch_size == 0:

                # open batch file and append to index
                batch_path = join(batches_dir, '{:d}.txt'.format(batch_id))
                index.write('{:s}\n'.format(relpath(batch_path, self.path)))
                batch_file = open(batch_path, 'w')

                # create log directory for batch
                mkdir(join(logs_dir, '{:d}'.format(batch_id)))

            # write paths to batch file
            batch_file.write('{:s}\n'.format(simulation_path))

            # close batch file
            if i % self.batch_size == (self.batch_size - 1):
                batch_file.close()
                chmod(batch_path, 0o755)

        index.close()

        chmod(index_path, 0o755)

    def make_directory(self):
        """ Create job subdirectory. """

        # create directory (overwrite existing one)
        path = join(self.sweep.path, 'benchmark')
        if not isdir(path):
            mkdir(path)

        # make subdirectories for simulations and scripts
        mkdir(join(path, 'scripts'))
        mkdir(join(path, 'batches'))
        mkdir(join(path, 'log'))






    def build(self,
              directory='./',
              save_history=True,
              walltime=10,
              allocation='p30653',
              cores=1,
              memory=4,
              **sim_kw):
        """
        Build job directory tree. Instantiates and saves a simulation instance for each parameter set, then generates a single shell script to submit each simulation as a separate job.

        Args:

            directory (str) - destination path

            save_history (bool) - if True, save simulation history

            walltime (int) - estimated job run time

            allocation (str) - project allocation

            cores (int) - number of cores per batch

            memory (int) - memory per batch, GB

            sim_kw (dict) - keyword arguments for simulation

        """

        # create job directory
        self.make_directory(directory)

        # store parameters (e.g. pulse conditions)
        self.sim_kw = sim_kw

        # build simulations
        for i, parameters in enumerate(self.parameters):
            simulation_path = join(self.path, 'simulations', '{:d}'.format(i))
            self.simulation_paths[i] = relpath(simulation_path, self.path)
            self.build_simulation(parameters, simulation_path, **sim_kw)

        # save serialized job
        with open(join(self.path, 'job.pkl'), 'wb') as file:
            pickle.dump(self, file, protocol=-1)

        # build parameter file for each batch
        self.build_batches()

        # build job run script
        self.build_run_script(self.path,
                              self.script_name,
                              save_history)

        # build job submission script
        self.build_submission_script(self.path,
                                     self.script_name,
                                     save_history,
                                     walltime=walltime,
                                     allocation=allocation,
                                     cores=cores,
                                     memory=memory)

    @classmethod
    def build_simulation(cls, parameters, simulation_path, **kwargs):
        """
        Builds and saves a simulation instance for a set of parameters.

        Args:

            parameters (iterable) - parameter sets

            simulation_path (str) - simulation path

            kwargs: keyword arguments for GrowthSimulation

        """

        # instantiate simulation
        simulation = GrowthSimulation(*parameters, **kwargs)

        # create simulation directory
        if not isdir(simulation_path):
            mkdir(simulation_path)

        # save simulation
        simulation.save(simulation_path)

    def load_simulation(self, index):
        """
        Load simulation instance from file.

        Args:

            index (int) - simulation index

        Returns:

            simulation (GrowthSimulation)

        """
        simulation_path = join(self.path, self.simulation_paths[index])
        return GrowthSimulation.load(simulation_path)






