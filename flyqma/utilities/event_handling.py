from distutils.util import strtobool


class UserPrompts:
    """
    Event handling for user prompts.
    """

    @classmethod
    def boolean_prompt(cls, question, num_attempts=0, hint=' [y/n] '):
        """ Prompt user for boolean response to <question>. """

        response = str(input(question + hint))

        try:
            if strtobool(response):
                return True
            else:
                return False
        except:
            print('Response not recognized, please try again.')
            if num_attempts >= 3:
                return False
            return cls.boolean_prompt(question, num_attempts+1, hint)

    @classmethod
    def integer_prompt(cls, question, num_attempts=0, hint=' [0-16] '):
        """ Prompt user for integer response to <question>. """

        response = str(input(question + hint))

        try:
            return int(response)

        except:
            print('Integer value required, please try again.')
            if num_attempts >= 3:
                return None
            return cls.integer_prompt(question, num_attempts+1, hint)
