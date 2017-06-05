import os
import numpy as np
from subprocess import call
import kenlm

class LanguageModel:
    '''
    interface for KenLM
    '''

    def __init__(self):
        self.model = None

    def kenlm_estimate(self, it, output_dir, kenlm_bin_dir):
        '''
        outputs all docs in it into a single file that can be read by KenLM
        runs KenLM estimator on this data

        writes more then one file into the specified directory!
        '''
        with open(os.path.join(output_dir, 'corpus.txt'), 'w') as f:
            for document in it.documents():
                f.write(' '.join(document.tokens) + '\n')

        call([os.path.join(kenlm_bin_dir, "lmplz"), "-o", "2",
              "--text", os.path.join(output_dir, 'corpus.txt'),
              "--arpa", os.path.join(output_dir, 'model.arpa')])
        call([os.path.join(kenlm_bin_dir, "build_binary"),
              os.path.join(output_dir, 'model.arpa'),
              os.path.join(output_dir, 'model.klm')])

    def load_model(self, model_dir):
        self.model = kenlm.LanguageModel(os.path.join(model_dir, 'model.klm'))

    def score_text(self, s):
        return self.model.score(s)

    def score_context(self, mention):
        return self.score_text(' '.join(mention.left_context() +
                               mention.mention_text_tokenized() +
                               mention.right_context()))

    def dataset_statistics(self, it, window_sz=20):
        '''
        Use language model to get some statistics for a dataset
        '''
        print 'scoring dataset according to LM'
        probs = [[] for i in xrange(window_sz*2 + 1)]
        maxx = [0.0 for i in xrange(window_sz*2 + 1)]
        txt = ['' for i in xrange(window_sz*2 + 1)]
        minn = [999.0 for i in xrange(window_sz*2 + 1)]
        txt2 = ['' for i in xrange(window_sz*2 + 1)]
        lengths = [0 for i in xrange(window_sz*2 + 1)]
        for i, mention in enumerate(it.mentions()):
            if i % 10 != 0:
                continue
            if i > 350000:
                break

            left_c = mention.left_context()[-window_sz:] \
                     if len(mention.left_context()) > window_sz else mention.left_context()
            right_c = mention.right_context()[0:window_sz] \
                     if len(mention.right_context()) > window_sz else mention.right_context()
            l = len(left_c) + len(right_c)
            lengths[l] += 1
            prob = math.exp(self.score_text(' '.join(left_c)) + self.score_text(' '.join(right_c)))
            probs[l].append(prob)
            if prob > maxx[l]:
                maxx[l] = prob
                txt[l] = left_c + ['!!!'] + right_c
            if prob < minn[l]:
                minn[l] = prob
                txt2[l] = left_c + ['!!!'] + right_c
            if i % 100000 == 0:
                print i, '...'
        for i in xrange(window_sz*2 + 1):
            if len(probs[i]) == 0:
                continue
            prob_mean = np.mean(probs[i])
            prob_var = np.var(probs[i])
            prob_median = np.median(probs[i])
            min_prob = min(probs[i])
            max_prob = max(probs[i])
            print i, 'mean context probability', prob_mean
            print i, 'variance:', prob_var, 'min:', min_prob, 'max:', max_prob, 'median:', prob_median
            print i, 'best:', txt[i]
            print i, 'worst:', txt2[i]
            print lengths

if __name__ == "__main__":
    from readers.conll_reader import CoNLLIterator
    from DbWrapper import *
    from WikilinksIterator import WikilinksNewIterator
    import math
    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
    _path = "/home/yotam/pythonWorkspace/deepProject"
    it = CoNLLIterator(_path + '/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', wikiDB, split='train')
    lm = LanguageModel()
    it = WikilinksNewIterator(_path+"/data/intralinks/all")
    lm.load_model(_path + '/data/LM')
    lm.dataset_statistics(it)
