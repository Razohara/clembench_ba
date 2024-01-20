"""Generation of Random Game Instances"""

import json
import random
import string
import os

from clemgame.clemgame import GameInstanceGenerator


# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results
GAME_NAME = 'grounding'
# we will create 10 instances for each experiment; vary this as you wish
N_INSTANCES = 10
# if the generation involves randomness, remember to set a random seed
SEED = 123

class GroundingGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        # always do this to initialise GameInstanceGenerator
        super().__init__(GAME_NAME)
    
    def load_file(self, path: str) -> dict:
        current_wd = os.getcwd()
        print(current_wd)
        return json.load(open(path))

    # define on_generate, a mandatory method
    def on_generate(self):
        # get the list of topics, which will be our experiments
        know_base = self.load_file('games/grounding/resources/knowledge_base.txt')
        # get the prompts for player a and player b
        # we'll keep the prompts fixed in all instances, replacing only the
        # necessary slots (but you can do it differently)
        prompt_a = self.load_template('resources/initial_prompts/initial_prompt_a')
        prompt_b = self.load_template('resources/initial_prompts/initial_prompt_b')

        # create experiment for each context
        for context, entities in know_base.items():
            experiment = self.add_experiment(context)
            
            # build N_INSTANCES instances for each experiment
            for game_id in range(N_INSTANCES):
                # create each instance with 2 random subjects from 
                # context entities and get their info
                subjects = random.sample(entities.keys(), 2)
                entity1_info = entities[subjects[0]]
                entity2_info = entities[subjects[1]]

                # for testing purposes 3 turns
                # TO DO: more turns outside of testing, random between 4 and 10?
                n_turns = 3

                # create a game instance, using a game_id counter/index
                instance = self.add_game_instance(experiment, game_id)

                # populate the game instance with its parameters
                instance['subjects'] = subjects
                instance['entity1_info'] = entity1_info
                instance['entity2_info'] = entity2_info
                instance['n_turns'] = n_turns
                instance['prompt_player_a'] = self.create_prompt(
                    context, subjects, entity1_info, entity2_info, n_turns, prompt_a)
                instance['prompt_player_b'] = self.create_prompt(
                    context, subjects, entity1_info, entity2_info, n_turns, prompt_b)
    
    # additional method, specifically for the grounding game's template
    def create_prompt(self,
                      context: str,
                      subjects: list,
                      entity1_info: list,
                      entity2_info: list,
                      n_turns: int,
                      prompt: str) -> str:
        """Replace a prompt template with slot values."""
        text = string.Template(prompt).substitute(context=context, subjects=subjects,
                                                  entity1_info=entity1_info,
                                                  entity2_info=entity2_info,
                                                  nturns=n_turns)
        return text


if __name__ == '__main__':
    random.seed(SEED)
    # always call this, which will actually generate and save the JSON file
    GroundingGameInstanceGenerator().generate()