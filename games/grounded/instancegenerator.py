"""
A game probing for common ground in LLMs.
Generation of grounded game instances to be played. 
"""

import json
import random
import string

from typing import List, Dict, Tuple

from clemgame.clemgame import GameInstanceGenerator
from games.grounded.constants import (
    GAME_NAME, LANG, N_INSTANCES, N_INFO_PER_EXP, N_TURNS_PER_DUR, SEED,
    REFLEXIVE, SHARED, WORDS_PATH, KNOW_PATH, PROMPT_PATH, DISCUSSEDQ)


class GroundedGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        # initialise GameInstanceGenerator
        super().__init__(GAME_NAME)
        words = self.load_json(WORDS_PATH.format(LANG))
        self.atag = words["ASIDE"]
        self.qtag = words["ME"]
        self.probe = words['PROBE']

    def load_file(self, path: str) -> dict:
        return json.load(open(path))

    def on_generate(self):
        """Generate instance."""

        # get knowledge base
        know_base = self.load_file(KNOW_PATH.format(LANG))

        # get prompts for each player
        prompt_a = self.load_template(PROMPT_PATH.format('a'))
        prompt_b = self.load_template(PROMPT_PATH.format('b'))

        # create experiments with differing n amount of turns
        for exp, n in N_TURNS_PER_DUR.items():
                experiment = self.add_experiment(exp)
                turns = n

                for game_id in range(N_INSTANCES):
                    # get information from knowledge base
                    # random fact per instance
                    info = random.sample(sorted(know_base.items()), 1)
                    fact = 'Simon ' + info[0][0]
                    fact_labels = info[0][1] # labels just to check fact mentioned later

                    # create a game instance, using a game_id counter/index
                    instance = self.add_game_instance(experiment, game_id)
                    
                    # populate the game instance with its parameters
                    instance['fact'] = fact
                    instance['fact_labels'] = fact_labels
                    instance['n_turns'] = turns
                    instance['prompt_player_a'] = self.create_prompt(fact,
                                                                     turns,
                                                                     prompt_a)
                    instance['prompt_player_b'] = self.create_prompt(fact,
                                                                     turns,
                                                                     prompt_b)
                    instance['probe_reflexive'] = self.probe.format(
                                                  REFLEXIVE.format(fact))
                    instance['probe_shared'] = self.probe.format(
                                               SHARED.format(fact))
                    instance['probe_discussed'] = self.probe.format(
                                                DISCUSSEDQ.format(fact))

    def create_prompt(self,
                      fact: str,
                      n_turns: int,
                      prompt: str) -> str:
        """Filling slots in prompt template (grounded specific)."""
        text = string.Template(prompt).substitute(fact=fact,
                                                  nturns=n_turns,
                                                  qtag=self.qtag,
                                                  atag=self.atag)
        return text


if __name__ == '__main__':
    random.seed(SEED)
    # generate and save JSON file containing instances
    GroundedGameInstanceGenerator().generate()
