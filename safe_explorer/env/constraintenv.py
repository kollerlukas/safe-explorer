import gym


class ConstraintEnv(gym.Env):

    def get_num_constraints(self):
        """
        return the number of constaints
        """
        pass

    def get_constraint_values(self):
        """
        return the constraint values as a list
        """
        pass

    def reset(self, random_agent_position=False):
        """
        return the constraint values as a list

        params:
          - random_agent_position: spawn agent in a random position
        """
        pass
