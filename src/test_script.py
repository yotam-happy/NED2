"""
Mannually execute each class dependency to work with the project without compiling
Best and easiest way to debug code in python. You can actully view to whole workspace

ATTENTION: Change the path variable in the end of the file
ATTENTION: If you de modificaitons in the classes please keep the copyies here updated! (There must be a better way but I am lazy)

NOTEs: use collapse all shortcut CTRL + SHIFT + NumPad (-)  to navigate and excecute easily the code.
also use the ALT+SHIFT+E to execute single lines or whole code fragments
I also recommend on Pycharm cell mode plugin for easier execution of code fragments
(Noam)
"""

## The cell seperator

from Evaluation import *
from PairwisePredict import *
from ModelTrainer import *
from models.DeepModel import *
from FeatureGenerator import *
from PointwisePredict import *
from PairwisePredict import *
from DbWrapper import WikipediaDbWrapper
from Candidates import *
##

def eval(experiment_name, path, train_session_nr, model, candidator, iter_eval, wordExclude=None, wordInclude=None,
         stats=None, sampling=None):
    # evaluate
    print "Evaluating " + experiment_name + "...", train_session_nr
    evaluation = Evaluation(iter_eval, model, candidator, wordExcludeFilter=wordExclude, wordIncludeFilter=wordInclude,
                            sampling=sampling, stats=stats)
    evaluation.evaluate()

    # save
    print "Saving...", train_session_nr
    precision_f = open(path + "/models/" + experiment_name + ".precision.txt", "a")
    precision_f.write(str(train_session_nr) + " train micro p@1: " + str(evaluation.mircoP()) +
                      ", macro p@1: " + str(evaluation.macroP()) + "\n")
    precision_f.close()


def experiment(experiment_name, path, model, train_stats, iter_train, iter_eval, filterWords = False,
               filterSenses = False, doEvaluation=True, p=1.0):
    '''
    :param experiment_name: all saved files start with this name
    :param path:            path to save files
    :param model:           model to train
    :param train_stats:     stats object of the training set
    :param iter_train:      iterator for the training set
    :param iter_eval:       iterator for the evaluation set
    :param filterWords:     True if we wish to treat 10% of the words as unseen (filter them from the training set)
    :param filterSenses:    True if we wish to treat 10% of the words, and all possible senses for them as unseen
                            (filter them from the training set)
    :param doEvaluation:    False if we want only training, without evaluation after every train session
    :param p:               fraction of words to train/test on (reduce the problem size)
    '''

    wordsForBroblem = train_stats.getRandomWordSubset(p)
    wordFilter = train_stats.getRandomWordSubset(0.1, baseSubset=wordsForBroblem) if filterWords or filterSenses \
        else None
    senseFilter = train_stats.getSensesFor(wordFilter) if filterSenses else None

    candidator = CandidatesUsingStatisticsObject(train_stats)
    trainer = ModelTrainer(iter_train, candidator, train_stats, model, epochs=1, neg_sample=1,
                           mention_include=wordsForBroblem, mention_exclude=wordFilter, sense_filter=senseFilter,
                           neg_sample_uniform=True, neg_sample_all_senses_prob=0.0)

    mps = 0
    t = 0
    kk = 0
    kkk = 0
    for w in wordsForBroblem:
        mm = 0
        for x, y in train_stats.mentionLinks[w].iteritems():
            if y > mm:
                mm = y
            t += y
        mps += mm
        kkk += len(train_stats.mentionLinks[w])
        kk += 1
    print "expected mps: ", float(mps) / t
    print "avg candidates per mention: ", float(kkk) / kk

    for train_session in xrange(500):
        # train
        print "Training... ", train_session
        trainer.train()

        model.saveModel(path + "/models/" + experiment_name + "." + str(train_session) + ".out")

        if doEvaluation:
            eval(experiment_name + ".eval", path, train_session, model, candidator, iter_eval,
                 wordInclude=wordsForBroblem, wordExclude=wordFilter, stats=train_stats, sampling=0.1)
            if filterWords or filterSenses:
                eval(experiment_name + ".unseen.eval", path, train_session, model, candidator, iter_eval,
                     wordInclude=wordFilter, stats=train_stats)

    # Plot train loss to file
    model.plotTrainLoss(path + "/models/" + experiment_name + ".train_loss.png", st=10)

##

_path = "/home/yotam/pythonWorkspace/deepProject"
print "Loading iterators+stats..."
if not os.path.isdir(_path):
    _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

# train on wikipedia intra-links corpus
#_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intralinks/train-stats")
#_iter_train = WikilinksNewIterator(_path+"/data/intralinks/filtered")
#_iter_eval = WikilinksNewIterator(_path+"/data/intralinks/filtered")

_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/filtered-train-stats")
_iter_train = WikilinksNewIterator(_path+"/data/wikilinks/filtered/train")
_iter_eval = WikilinksNewIterator(_path+"/data/wikilinks/filtered/validation")
print "Done!"

print 'Loading embeddings...'
_w2v = Word2vecLoader(wordsFilePath=_path+"/data/word2vec/dim300vecs",
                      conceptsFilePath=_path+"/data/word2vec/dim300context_vecs")
wD = _train_stats.contextDictionary
cD = {int(x) for x in _train_stats.conceptCounts}
_w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
#_w2v.randomEmbeddings(wordDict=wD, conceptDict=cD)
print 'wordEmbedding dict size: ', len(_w2v.wordEmbeddings), " wanted: ", len(wD)
print 'conceptEmbeddings dict size: ', len(_w2v.conceptEmbeddings), " wanted", len(cD)
print 'Done!'

print 'Connecting to db'
wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002',
                            concept_filter=_w2v.conceptDict)
print 'Done!'

# TRAIN PAIRWISE MODEL
print 'Training...'

model = DeepModel(_path + '/models/basic_model.config', w2v=_w2v, stats=_train_stats, db=wikiDB)
experiment("small", _path, model, _train_stats, _iter_train, _iter_eval,
           doEvaluation=True, filterWords=False)

## baseline
#_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/train-stats")
#_iter_test = WikilinksNewIterator(_path+"/data/wikilinks/all/evaluation")
#_pairwise_model = BaselinePairwiseModel(_train_stats)
#_pairwise_model = GuessPairwiseModel()
#predictor = PairwisePredict(_pairwise_model, _train_stats)
#evaluation = Evaluation(_iter_test, predictor, stats=_train_stats)
#evaluation.evaluate()

