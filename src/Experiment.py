import ProjectSettings
from Candidates import *
from DbWrapper import WikipediaDbWrapper
from DbWrapperCached import *
from ModelTrainer import ModelTrainer
from PPRforNED import PPRStatistics
from WikilinksStatistics import WikilinksStatistics
from Word2vecLoader import Word2vecLoader
from readers.conll_reader import CoNLLIterator
from GBRTmodel import GBRTModel
from WikilinksIterator import *
from Evaluation import *
from models.DeepModel import *
from EntityTransform import *

class Experiment:
    def __init__(self, config, load_path=None):
        self.path, self.pc_name = ProjectSettings.getPath()

        if type(config) == str:
            with open(self.path + config) as data_file:
                self._config = json.load(data_file)
        else:
            self._config = config

        if load_path is not None:
            self._config['model']['load_path'] = load_path

        self.db = self.connect_db(self._config['db'])

        self.entity_transforms = dict()
        if "entity_transforms" in self._config:
            for x in self._config["entity_transforms"]:
                self.entity_transforms[x["name"]] = self.switch_entity_transform(x)

        self.iterators = dict()
        for x in self._config["iterators"]:
            self.iterators[x["name"]] = self.switch_iterator(x)

        self.stats = dict()
        for x in self._config["stats"]:
            self.stats[x["name"]] = self.load_stats(x)

        self.candidator = self.switch_candidator(self._config['candidator'])
        self.w2v = self.load_w2v(self._config['w2v']) if 'w2v' in self._config else None

        self.model = self.switch_model(self._config['model'])
        self.trained_mentions = None

        # TODO: somehow fix this serious hack??
        if hasattr(self.model, '_feature_generator') \
                and hasattr(self.model._feature_generator, 'yamada_txt_to_embd') \
                and self.model._feature_generator.yamada_txt_to_embd is not None:
            self.candidator._filter = self.model._feature_generator.yamada_txt_to_embd

    def switch_model(self, config):
        models_as_features = dict()
        if 'models_as_features' in config:
            for x in config['models_as_features']:
                models_as_features[x['name']] = self.switch_model(x)

        print "loading model..."
        if config["type"] == 'deep_model':
            transform = self.entity_transforms[config['entity_transform']] if 'entity_transform' in config else None
            return DeepModel(self.path + config['config_path'],
                             w2v=self.w2v,
                             stats=self.stats[config['stats']],
                             db=self.db,
                             load_path=self.path + str(config['load_path']) if 'load_path' in config else None,
                             models_as_features=models_as_features,
                             entity_transform=transform,
                             inplace_transform=config['inplace_transform'] if 'inplace_transform' in config else False)
        elif config["type"] == 'gbrt':
            return GBRTModel(self.path + config['config_path'],
                             db=self.db,
                             stats=self.stats[config['stats']],
                             models_as_features=models_as_features,
                             load_path=self.path + str(config['load_path']) if 'load_path' in config else None,
                             w2v=self.w2v)
        else:
            raise "Config error"

    def load_w2v(self, config):
        print "loading w2v..."
        w2v = Word2vecLoader(wordsFilePath=self.path + config['words_path'],
                             conceptsFilePath=self.path + config['concepts_path'])
        concept_filter = self.switch_concept_filter(config['concept_filter']) if 'concept_filter' in config else None
        if 'random' in config and config['random']:
            w2v.randomEmbeddings(conceptDict=concept_filter)
        else:
            w2v.loadEmbeddings(conceptDict=concept_filter)
        return w2v

    def switch_concept_filter(self, config):
        if config['src'] == 'by_iter':
            return self.candidator.getAllCandidateSet(self.iterators[config['iterator']])
        elif config['src'] == 'by_stats':
            return {int(x) for x in self.stats[config['stats']].conceptCounts}
        else:
            raise "Config error"

    def switch_entity_transform(self, config):
        if config['type'] == 'EntityToTypeByDbPediaLhdDataset':
            return EntityToTypeByDbPediaLhdDataset(self.path + config['ttl_path'], self.db, int(config['embd_sz']))

    def load_stats(self, config):
        print "loading statistics:", config["name"]
        if config['src'] == 'stats_file':
            transform = self.entity_transforms[config['entity_transform']] if 'entity_transform' in config else None
            return WikilinksStatistics(None, load_from_file_path=self.path + config['path'],
                                       entity_transform=transform)
        elif config['src'] == 'ppr':
            return PPRStatistics(None, self.path + config['path'],
                                 fill_in=self.stats[config['fillin']] if 'fillin' in config else None)
        else:
            raise "Config error"

    def connect_db(self, config):
        print "connecting to db..."
        if 'cached' in config and config['cached']:
            print "load cache"
            db = WikipediaDbWrapperCached()
            db.load(self.path + config['path'])
            return db
        else:
            return WikipediaDbWrapper(user=config['user'],
                                       password=config['password'],
                                       database=config['database'])

    def switch_iterator(self, config):
        if config['dataset'] == 'conll':
            return CoNLLIterator(self.path + '/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', self.db, split=config['split'])
        elif config['dataset'] == 'from_json':
            return WikilinksNewIterator(self.path + config['path'])
        else:
            raise "Config error"

    def switch_candidator(self, config):
        handler = CandidatesHandler()
        if 'entity_transform' in config:
            handler.entity_transform = self.entity_transforms[config['entity_transform']]
        if 'disambig_filter' in config:
            handler.candidate_filter = CandidateDisambigPageFilter(self.db)

        if config['type'] == 'yago':
            handler.candidate_source = CandidatesUsingYago2(self.path + config['path'])
        elif config['type'] == 'ppr':
            handler.candidate_source = CandidatesUsingPPRStats(self.stats[config['stats']], self.db)
        elif config['type'] == 'stats':
            handler.candidate_source = CandidatesUsingStatisticsObject(self.stats[config['stats']], db=self.db)
        else:
            raise "Config error"
        return handler

    def train(self, config=None, model_name=None):
        if config is None:
            config = self._config["training"]
        if 'train' in config and not config['train']:
            return

        print 'beging training...'
        transform = self.entity_transforms[config['entity_transform']] if 'entity_transform' in config else None
        trainer = ModelTrainer(self.iterators[config['iterator']],
                               self.candidator,
                               self.stats[config['stats']],
                               self.model,
                               epochs=config['epochs'],
                               neg_sample=config['neg_samples'],
                               neg_sample_uniform=config['neg_sample_uniform'],
                               neg_sample_all_senses_prob=config['neg_sample_all_senses_prob'],
                               sampling=config['sampling'] if 'sampling' in config else None,
                               subsampling=config['subsampling'] if 'subsampling' in config else None,
                               entity_transform=transform)
        self.trained_mentions = trainer.train()
        path = self.path + self._config['model']['config_path'] + (model_name if model_name is not None else "")
        self.model.saveModel(path)
        print 'Done!'

    def evaluate(self, config=None):
        if config is None:
            config = self._config["evaluation"]
        transform = self.entity_transforms[config['entity_transform']] if 'entity_transform' in config else None
        evaluation = Evaluation(self.iterators[config['iterator']],
                                self.model,
                                self.candidator,
                                self.stats[config['stats']],
                                sampling=config['sampling'] if 'sampling' in config else None,
                                log_path=self.path + "/evaluation.txt",
                                db=self.db,
                                trained_mentions=self.trained_mentions,
                                attn=config['attn'] if 'attn' in config else None,
                                entity_transform=transform)
        evaluation.evaluate()

if __name__ == "__main__":
    experiment = Experiment("/experiments/zz_wikilinks_deep_entities/experiment.conf")
    for x in xrange(8):
        trained_mentions = experiment.train(model_name='.'+str(x))
        experiment.evaluate()

