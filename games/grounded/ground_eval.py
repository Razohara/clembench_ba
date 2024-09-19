"""
A game to test the extent of common ground of models.
Evaluation of common ground probes taken throughout the game. 
"""

import re
import string
import collections
import gensim.downloader as api
from nltk.tokenize import RegexpTokenizer
from evaluate import load
from sentence_transformers import SentenceTransformer, util


class ProbeEval:
    """Evaluation of Grounding Game Probes"""

    def __init__(self, episode_probe_responses_a: list[str],
                 episode_probe_responses_b: list[str], mcp=False):
        """Get probe responses and sentence transformer model."""
        self.mcp = mcp
        self.responses_a = episode_probe_responses_a
        self.responses_b = episode_probe_responses_b

        if not self.mcp:
            # lower
            for i in range(len(self.responses_a)):
                self.responses_a[i] = ' '.join(word.lower()
                                               for word in self.responses_a[i].split())
                self.responses_b[i] = ' '.join(word.lower()
                                               for word in self.responses_b[i].split())

        # tokenize & remove punct
        tok = RegexpTokenizer(r'\w+')
        self.tok_responses_a = tok.tokenize_sents(self.responses_a)
        self.tok_responses_b = tok.tokenize_sents(self.responses_b)

        # collect results
        self.results = dict()

        # load pretrained model for vector embeddings using sentence transformer
        self.model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

        # switch for incorrect mcp format e.g. through abortions
        self.aborted = False

        # check for list of strings containing numbers from mcp responses
        # and remove invalid responses from retries during mcp handling
        if self.mcp:
            self.num_tok_responses_a = self.tok_responses_a
            self.num_tok_responses_b = self.tok_responses_b
            for ls1 in self.num_tok_responses_a[:]:
                if not all(isinstance(item, str) and bool(re.match(r'^\d+$', item)) for item in ls1):
                    print(self.num_tok_responses_a)
                    print(ls1)
                    self.num_tok_responses_a.remove(ls1)
                    print(self.num_tok_responses_a)
            for ls2 in self.num_tok_responses_b[:]:
                if not all(isinstance(item, str) and bool(re.match(r'^\d+$', item)) for item in ls2):
                    print(self.num_tok_responses_b)
                    print(ls2)
                    self.num_tok_responses_b.remove(ls2)
                    print(self.num_tok_responses_b)

            # convert strings to numeric
            self.num_tok_responses_a = [[int(string) for string in ele]
                                        for ele in self.num_tok_responses_a]
            self.num_tok_responses_b = [[int(string) for string in ele]
                                        for ele in self.num_tok_responses_b]

            # check for one sided abort so unequal length of responses
            if len(self.num_tok_responses_a) != len(self.num_tok_responses_b):
                print('onesided abort')
                longer_list = self.num_tok_responses_a if len(self.num_tok_responses_a) > len(self.num_tok_responses_b) \
                    else self.num_tok_responses_b
                longer_list.pop()

            # check for initial one sided or initial two sided abort
            if not self.num_tok_responses_a or not self.num_tok_responses_b:
                print('one sided abort was initial or initial two sided')
                self.aborted = True

            print(self.num_tok_responses_a)
            print(self.num_tok_responses_b)

    def _compute_f1overlap(self):
        """Compute overlap using the f1 score for word overlap between
        probe responses of players, used for surface similarity
        (Stanford Question Answering Dataset (SQuAD))."""
        f1_overlaps_per_turn = []

        for i in range(len(self.tok_responses_a)):
            overlap = collections.Counter(self.tok_responses_a[i]) & \
                      collections.Counter(self.tok_responses_b[i])
            sum_overlap = sum(overlap.values())

            if sum_overlap == 0:
                f1_overlaps_per_turn.append(0)
                continue

            if len(self.tok_responses_a[i]) == 0 or \
               len(self.tok_responses_b[i]) == 0:
                f1_overlaps_per_turn.append(int(self.tok_responses_a[i] ==
                                                self.tok_responses_b[i]))
                continue

            precision = 1.0 * sum_overlap / len(self.tok_responses_a[i])
            recall = 1.0 * sum_overlap / len(self.tok_responses_b[i])
            f1_overlaps_per_turn.append(
                round(((2 * precision * recall) / (precision + recall)), 4))

        self.results['F1 overlap'] = {'final': round(f1_overlaps_per_turn[-1], 4),
                                      'avg': round(sum(f1_overlaps_per_turn) /
                                                   len(f1_overlaps_per_turn), 4),
                                      'per turn': f1_overlaps_per_turn}

    def _compute_bleu(self):
        """Compute BLEU Score for similarity of probe responses of players,
        used for surface similarity (hugging face module)."""
        bleu_per_turn = []
        bleu = load('bleu')

        for tup in zip(self.responses_a, self.responses_b):
            results = bleu.compute(predictions=[tup[0]], references=[
                                   tup[1]], smooth=True)
            bleu_per_turn.append(round(results['bleu'], 4))

        self.results['BLEU'] = {'final': round(bleu_per_turn[-1], 4),
                                'avg': round(sum(bleu_per_turn) /
                                             len(bleu_per_turn), 4),
                                'per turn': bleu_per_turn}

    def _compute_rouge(self):
        """Compute ROGUE Score for similarity of probe responses of players,
        used for surface similarity (hugging face module)."""
        rouge1_per_turn = []
        rouge2_per_turn = []
        rougeL_per_turn = []
        rougeLsum_per_turn = []
        rouge = load('rouge')

        for tup in zip(self.responses_a, self.responses_b):
            results = rouge.compute(predictions=[tup[0]], references=[tup[1]])
            rouge1_per_turn.append(round(results['rouge1'], 4))
            rouge2_per_turn.append(round(results['rouge2'], 4))
            rougeL_per_turn.append(round(results['rougeL'], 4))
            rougeLsum_per_turn.append(round(results['rougeLsum'], 4))

        self.results['ROUGE1'] = {'final': round(rouge1_per_turn[-1], 4),
                                  'avg': round(sum(rouge1_per_turn) /
                                               len(rouge1_per_turn), 4),
                                  'per turn': rouge1_per_turn}
        self.results['ROUGE2'] = {'final': round(rouge2_per_turn[-1], 4),
                                  'avg': round(sum(rouge2_per_turn) /
                                               len(rouge2_per_turn), 4),
                                  'per turn': rouge2_per_turn}
        self.results['ROUGEL'] = {'final': round(rougeL_per_turn[-1], 4),
                                  'avg': round(sum(rougeL_per_turn) /
                                               len(rougeL_per_turn), 4),
                                  'per turn': rougeL_per_turn}
        self.results['ROUGELsum'] = {'final': round(rougeLsum_per_turn[-1], 4),
                                     'avg': round(sum(rougeLsum_per_turn) /
                                                  len(rougeLsum_per_turn), 4),
                                     'per turn': rougeLsum_per_turn}

    def _compute_meteor(self):
        """Compute METEOR Score for similarity of probe responses of players,
        used for surface as well as semantic similarity (hugging face module)."""
        meteor_per_turn = []
        meteor = load('meteor')

        for tup in zip(self.responses_a, self.responses_b):
            results = meteor.compute(predictions=[tup[0]], references=[tup[1]])
            meteor_per_turn.append(round(results['meteor'], 4))

        self.results['METEOR'] = {'final': round(meteor_per_turn[-1], 4),
                                  'avg': round(sum(meteor_per_turn) /
                                               len(meteor_per_turn), 4),
                                  'per turn': meteor_per_turn}

    def _compute_word_mover(self):
        """Compute Word Mover Distance for similarity of probe responses of players,
        used for semantic similarity."""
        wm_per_turn = []
        model = api.load('word2vec-google-news-300')

        for tup in zip(self.tok_responses_a, self.tok_responses_b):
            result = model.wmdistance(tup[0], tup[1])
            wm_per_turn.append(round(result, 4))

        self.results['Word mover distance'] = {'final': round(wm_per_turn[-1], 4),
                                               'avg': round(sum(wm_per_turn) /
                                                            len(wm_per_turn), 4),
                                               'per turn': wm_per_turn}

    def _compute_vec_cos(self):
        """Compute Cosine Similarity of probe responses of players
        using vector representations, used for semantic similarity."""
        vec_cos_per_turn = []

        # sentence embeddings
        for i in range(len(self.responses_a)):
            embeddings = self.model.encode([self.responses_a[i],
                                            self.responses_b[i]])

            # cosine similarity
            cos = util.pytorch_cos_sim(embeddings[0],
                                       embeddings[1]).item()
            vec_cos_per_turn.append(round(cos, 4))

        self.results['Vector cosine sim'] = {'final': round(vec_cos_per_turn[-1], 4),
                                             'avg': round(sum(vec_cos_per_turn) /
                                                          len(vec_cos_per_turn), 4),
                                             'per turn': vec_cos_per_turn}

    def _compute_jacc_sim(self):
        """Compute intersection of multiple choice probe responses using Jaccard similarity."""
        jaccards_per_turn = []

        if self.aborted:
            return

        for tup in zip(self.num_tok_responses_a, self.num_tok_responses_b):
            # avoiding zero division
            if len(set(tup[0]).union(set(tup[1]))) == 0:
                ratio = 0
            else:
                intersect = set(tup[0]).intersection(set(tup[1]))

                # unique elements
                total_uniques = len(set(tup[0]).union(set(tup[1])))
                ratio = round(len(intersect) / total_uniques, 4)

            jaccards_per_turn.append(round(ratio, 4))

        self.results['Jaccard sim'] = {'final': round(jaccards_per_turn[-1], 4),
                                       'avg': round(sum(jaccards_per_turn) /
                                                    len(jaccards_per_turn), 4),
                                       'per turn': jaccards_per_turn}

    def _compute_overlap_coeff(self):
        """Compute overlap coefficient of multiple choice probe responses."""
        overlaps_per_turn = []

        if self.aborted:
            return

        for tup in zip(self.num_tok_responses_a, self.num_tok_responses_b):
            # avoiding zero division
            if min(len(set(tup[0])), len(set(tup[1]))) == 0:
                overlap = 0
            else:
                intersect = set(tup[0]).intersection(set(tup[1]))
                overlap = round((len(intersect) /
                                 min(len(set(tup[0])),
                                     len(set(tup[1])))), 4)

            overlaps_per_turn.append(round(overlap, 4))

        self.results['Overlap coeff'] = {'final': round(overlaps_per_turn[-1], 4),
                                         'avg': round(sum(overlaps_per_turn) /
                                                      len(overlaps_per_turn), 4),
                                         'per turn': overlaps_per_turn}

    def get_results(self):
        """Getter for evaluation results of probe responses (free or mcp)."""
        # return empty results dict in case of abortions
        if self.aborted:
            return self.results

        if self.mcp:
            self._compute_overlap_coeff()
            self._compute_jacc_sim()

        else:
            self._compute_f1overlap()
            self._compute_bleu()
            self._compute_rouge()
            self._compute_meteor()
            self._compute_word_mover()
            self._compute_vec_cos()

        return self.results


if __name__ == "__main__":
    # Tests considering syntactic and semantic aspects
    print('**************')
    same_a = ['Bicycles are very very very very very very fun.']
    same_b = ['Bicycles are very very very very very very fun.']
    words_same = ProbeEval(same_a, same_b)
    print('--------------')
    print('Same wording, same semantics.')
    print(same_a, same_b)
    print('--------------')
    for item in words_same.get_results().items():
        print(item)

    print('**************')
    same_opp_a = ['Bicycles are fun.']
    same_opp_b = ['Bicycles are boring.']
    words_same_mean_diff = ProbeEval(same_opp_a, same_opp_b)
    print('--------------')
    print('Same wording, opposite semantics.')
    print(same_opp_a, same_opp_b)
    print('--------------')
    for item in words_same_mean_diff.get_results().items():
        print(item)

    print('**************')
    similar_a = ['Bicycles are fun.']
    similar_b = ['Bikes bring joy.']
    words_diff_mean_same = ProbeEval(similar_a, similar_b)
    print('--------------')
    print('Different wording, same semantics.')
    print(similar_a, similar_b)
    print('--------------')
    for item in words_diff_mean_same.get_results().items():
        print(item)

    print('**************')
    similar_opp_a = ['Bicycles are boring.']
    similar_opp_b = ['Bikes bring joy.']
    words_diff_mean_diff = ProbeEval(similar_opp_a, similar_opp_b)
    print('--------------')
    print('Different wording, opposite semantics.')
    print(similar_opp_a, similar_opp_b)
    print('--------------')
    for item in words_diff_mean_diff.get_results().items():
        print(item)

    print('**************')
    random_a = ['Bikes are fun.']
    random_b = ['Cake contains sugar.']
    words_diff_mean_none = ProbeEval(random_a, random_b)
    print('--------------')
    print('Different wording, no semantics.')
    print(random_a, random_b)
    print('--------------')
    for item in words_diff_mean_none.get_results().items():
        print(item)

    # Tests with actual model responses
    print('**************')
    responses_a = ["It seems that we both know Peter and Katherine, and we are aware of some recent events in their lives, such as Peter's divorce and move, and Katherine's car sale and love for her job.", "As common ground in this conversation, we both know Peter and Katherine, and we have discussed some of their recent life events, such as Peter's divorce and move, and Katherine's car sale and love for her job.",
                   "As common ground in this conversation, we both know Peter and Katherine, and we have discussed their recent life events such as Peter's divorce and move, his allergy to cats, Katherine selling her car, and her love for her job and spending summers in France."]
    responses_b = ["The common ground in this conversation is our knowledge of Peter and Katherine's recent life events, such as Peter's divorce and move, and Katherine's car sale and love for her job.", "In this conversation, the common ground seems to be our mutual knowledge and discussion about Peter's recent life changes, such as his divorce, move, and allergy to cats.",
                   "In this conversation, the common ground seems to be our mutual knowledge and discussion about Peter and Katherine's recent life events, such as Peter's divorce and move, his allergy to cats, Katherine selling her car, and her love for her job and summers in France."]
    models_responses = ProbeEval(responses_a, responses_b)
    print('--------------')
    print('Tests with actual model responses')
    print('--------------')
    for item in models_responses.get_results().items():
        print(item)

    # Tests with actual model responses - mcq
    print('**************')
    # everything went fine
    resp_num_a = ['4, 5, 6']
    resp_num_b = ['1, 5']
    # resp_num_a = ['1, 2, 3, 4, 5, 6', '1, 4, 5, 6', '1, 4, 5, 6']
    # resp_num_b = ['1, 2, 3, 4, 5, 6', '1, 4, 5, 6', '1, 2, 3, 4, 5, 6']
    # onesided abort, but still response from other player
    # resp_num_a = ['2, 5', '3, 6']
    # resp_num_b = ['2, 3', 'number 1 and number 2', 'number 3 and number 4', 'number 5 and number 6']
    # onesided abort, but still response from other player, afterwards empty because at start
    # resp_num_a = ['1, 5']
    # resp_num_b = ['number 1 and number 2', 'number 3 and number 4', 'number 5 and number 6']
    # two sided abort, afterwards both empty because at start
    # resp_num_a = ['number 1 and number 2', 'number 3 and number 4', 'number 5 and number 6']
    # resp_num_b = ['number 10 and number 20', 'number 30 and number 40', 'number 50 and number 60']
    # no aborts, just retries, same responses
    # resp_num_a = ['1, 2, 3, 4, 5, 6', 'number 1 and number 2', 'number 1 and number 2', '1, 4, 5, 6']
    # resp_num_b = ['number 1 and number 2', '1, 2, 3, 4, 5, 6', '1, 4, 5, 6']
    models_responses_mcp = ProbeEval(resp_num_a, resp_num_b, mcp=True)
    print('--------------')
    print('Tests with actual model responses - mcq')
    print('--------------')
    for item in models_responses_mcp.get_results().items():
        print(item)
    print('**************')
