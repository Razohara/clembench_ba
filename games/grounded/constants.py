"""
Constants used in the grounded game and instance generator.
"""

import numpy as np


# game specifics
GAME_NAME = 'grounded'
LANG = 'en'
N_INSTANCES = 10
N_INFO_PER_EXP = {'small': 1, 'medium': 3, 'large': 5}
N_TURNS_PER_DUR = {'short': 5, 'moderate': 10, 'long': 15}
N_RETRIES = 3
N_PROBINGS = 10
N_PROBINGS_FINAL = 4

# paths
WORDS_PATH = 'resources/{}_words.json'
KNOW_PATH = 'games/grounded/resources/{}_knowledge_base.txt'
PROMPT_PATH = 'resources/initial_prompts/initial_prompt_{}'

# random seed for random generation
SEED = 123

# labels
INVALID = 'NA'
INVALID_LABEL = 2

# probing questions
REFLEXIVE = 'Do you know that {}?'
SHARED = 'Does your conversation partner know that {}?'
DISCUSSEDQ = 'Have you and your conversation partner discussed that {}?'

# expectations during probing
EXPECTATIONS = {'a': {'pre': {'refl': ['yes'], 
                              'symm': ['no', "don't know"]},
                      'final': {'discussed': 
                                {'refl': ['yes'], 
                                 'symm': ['yes']}, 
                                'not_discussed': 
                                {'refl': ['yes'], 
                                 'symm': ['no', "don't know"]}}}, 
                'b': {'pre': {'refl': ['no', "don't know"], 
                              'symm': ["don't know"]},
                      'final': {'discussed': 
                                {'refl': ['yes'], 
                                 'symm': ['yes']}, 
                                'not_discussed': 
                                {'refl': ['no', "don't know"], 
                                 'symm': ['no', "don't know"]}}}}
# standard messages
UPDATE = 'Value for {} anticipated; ground truth turn updated from {} to {}.'
NOT_SUCCESS = 'Answer for {} invalid after max attempts.'
SUCCESS = 'Answer for {} valid after {} tries.'
RESULT = 'Answer is {}correct.'
NOT_PARSED = 'Answer could not be parsed!'
