from os import getcwd
from argparse import ArgumentParser, ArgumentTypeError


# ======================== ARGUMENT TYPE CASTING ==============================

def str2bool(arg):
     """ Convert <arg> to boolean. """
     if arg.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
     elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
     else:
          raise ArgumentTypeError('Boolean value expected.')


# ======================== ARGUMENT PARSING ===================================


class RunArguments(ArgumentParser):
     """ Argument handler for run scripts. """

     def __init__(self, **kwargs):
          super().__init__(**kwargs)
          self.add_arguments()
          self.parse()

     def __getitem__(self, key):
          """ Returns <key> argument value. """
          return self.args[key]

     def add_arguments(self):
          """ Add arguments. """

          # add position argument for path
          self.add_argument(
               'path',
               nargs='?',
               default=getcwd())

          # add keyword argument for saving measurement data
          self.add_argument(
               '-S', '--save_data',
               help='Save measurement (training) data.',
               type=str2bool,
               default=False,
               required=False)

          # add keyword argument for globally training annotator
          self.add_argument(
               '-G', '--train_globally',
               help='Train global annotator.',
               type=str2bool,
               default=True,
               required=False)

     def parse(self):
          """ Parse arguments. """
          self.args = vars(self.parse_args())


class SweepArguments(RunArguments):
     """ Argument handler for parameter sweeps. """

     def add_arguments(self):
          """ Add arguments. """

          super().add_arguments()

          # add keyword argument for minimum fluorescence ambiguity
          self.add_argument('--min_ambiguity',
                              help='Minimum fluorescence ambiguity.',
                              type=float,
                              default=0.1,
                              required=False)

          # add keyword argument for maximum fluorescence ambiguity
          self.add_argument('--max_ambiguity',
                              help='Maximum fluorescence ambiguity.',
                              type=float,
                              default=1.,
                              required=False)

          # add keyword argument for number of fluorescence ambiguities
          self.add_argument('-a', '--num_ambiguities',
                              help='Number of fluorescence ambiguities.',
                              type=int,
                              default=16,
                              required=False)

          # add keyword argument for number of fluorescence replicates
          self.add_argument('-n', '--num_replicates',
                              help='Fluorescence replicates per simulation.',
                              type=int,
                              default=1,
                              required=False)

          # add keyword argument for sweep density
          self.add_argument('-lr', '--logratio',
                              help='Weight edges by logratio.',
                              type=str2bool,
                              default=True,
                              required=False)

          # add keyword argument for estimated run time
          self.add_argument('-w', '--walltime',
                              help='Estimated run time.',
                              type=int,
                              default=24,
                              required=False)

          # add keyword argument for number of cores
          self.add_argument('-c', '--cores',
                              help='Number of cores.',
                              type=int,
                              default=4,
                              required=False)

          # add keyword argument for memory usage
          self.add_argument('-m', '--memory',
                              help='Memory usage.',
                              type=int,
                              default=4,
                              required=False)

          # add keyword argument for project allocation
          self.add_argument('-A', '--allocation',
                              help='Project allocation.',
                              type=str,
                              default='p30653',
                              required=False)
