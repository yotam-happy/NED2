import utils.text
import json
from utils.document import *
import utils.read_dbpedia_ttl as ttl
import os

class CandidatesHandler:
    def __init__(self):
        self.candidate_source = None
        self.entity_transform = None
        self.candidate_filter = None

    def get_candidates_for_mention(self, mention):
        candidates = self.candidate_source.get_candidates_for_mention(mention)
        if self.candidate_filter is not None:
            candidates = self.candidate_filter.filter_candidates(candidates)
        #if self.entity_transform is not None:
        #    candidates = {self.entity_transform.entity_id_transform(x) for x in candidates}
        return candidates

    def add_candidates_to_mention(self, mention):
        mention.candidates = self.get_candidates_for_mention(mention)

    def add_candidates_to_document(self, document):
        for mention in document.mentions:
            self.add_candidates_to_mention(mention)

    def getAllCandidateSet(self, it):
        all_cands = set()
        for mention in it.mentions():
            c = self.get_candidates_for_mention(mention)
            for x in c:
                all_cands.add(x)
        return all_cands


class CandidatesUsingPPRStats:
    def __init__(self, pprstats, db):
        self._pprstats = pprstats
        self._db = db

    def get_candidates_for_mention(self, mention):
        candidates = set()

        if hasattr(self, '_filter'):
            embd = self._filter.into_the_cache(mention.gold_sense_id(),
                                               mention.gold_sense_url(),
                                               self._pprstats.conceptNames[mention.gold_sense_id()]
                                               if mention.gold_sense_id() in self._pprstats.conceptNames else None,
                                               verbose=True)

        for url in self._pprstats.getCandidateUrlsForMention(mention):
            page_id = self._db.resolvePage(url[url.rfind('/') + 1:])

            if page_id is not None:
                candidates.add(page_id)
                if hasattr(self, '_filter'):
                    embd = self._filter.into_the_cache(page_id, url, self._pprstats.conceptNames[page_id])

        return candidates


class CandidatesUsingStatisticsObject:
    def __init__(self, stats, db=None):
        self._stats = stats
        self._db = db

    def get_candidates_for_mention(self, mention):
        cands = self._stats.getCandidatesForMention(mention)

        if hasattr(self, '_filter'):
            # need to put yamada embd in cache
            title = self._db.getPageTitle(mention.gold_sense_id())
            embd = self._filter.into_the_cache(mention.gold_sense_id(),
                                               mention.gold_sense_url(),
                                               title,
                                               verbose=False)
            for c in cands:
                title = self._db.getPageTitle(c)
                embd = self._filter.into_the_cache(c,
                                                   title,
                                                   title,
                                                   verbose=False)

        return cands


class CandidatesUsingYago2:
    def __init__(self, path=None):
        self.mentions = dict()
        if path is not None:
            self.load(path)

    def import_yago(self, f, db):
        self._cache = dict()

        self.mentions = dict()
        self._import_yago2_file(f, db)

        # replace sets with arrays
        for mention, candidates in self.mentions.iteritems():
            arr = [x for x in candidates]
            self.mentions[mention] = arr

    def _import_yago2_file(self, path, db):
        k = 0
        t = 0
        with open(path) as f:
            print 'importing from path:', path
            for i, line in enumerate(f):
                if i % 10000 == 0 and t > 0:
                    print "done", i, "rows (", float(k) / t, " resolved)"

                tokens = line.split('\t')
                t += 1

                mention = utils.text.normalize_unicode(tokens[0].decode('unicode-escape'))
                mention = mention[1:-1]
                ent = tokens[1].decode('unicode-escape')
                ent = ent[:-1]
                entity_id = db.resolvePage(ent)

                if entity_id is not None:
                    if mention not in self.mentions:
                        self.mentions[mention] = dict()
                    k += 1
                    self.mentions[mention][entity_id] = 1

    def save(self, path):
        f = open(path, mode='w')
        f.write(json.dumps(self.mentions)+'\n')

    def load(self, path):
        """ loads statistics from a file """
        f = open(path, mode='r')
        l = f.readlines()
        self.mentions = json.loads(l[0])

    def get_candidates_for_mention(self, mention):
        mention = utils.text.normalize_unicode(mention.mention_text())
        return {int(x) for x in self.mentions[mention]} if mention in self.mentions else set()


# Filters disambiguation pages using dbpedia disambiguation page dataset (ttl format)
class CandidateDisambigPageFilter:
    def __init__(self, db, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.disambigs = pickle.load(f)
        else:
            print "One-time disambiguation pages resolution. This might take some minutes..."
            self.disambigs = db.getPagesForCategory("Disambiguation_pages")

    def filter_candidates(self, candidates):
        cc = set()
        for cand in candidates:
            if cand not in self.is_disambig:
                title = self.db.getPageTitle(cand)
                self.is_disambig[cand] = True if title.lower().find('disambig') >= 0 else False
            if not self.is_disambig[cand]:
                cc.add(cand)
        return cc


# Filters disambiguation pages using dbpedia disambiguation page dataset (ttl format)
# -- the db-pedia dataset was found to be high recall but too low precision!!
# -- (it mistakes a non-neglible number of pages as disambiguation)
class CandidateDisambigPageWithDbPediaFilter:
    def __init__(self, path, db):
        if os.path.isfile(path + '.id_cache'):
            self.disambig_page_ids = pickle.load(open(path + '.id_cache', 'r'))
            print 'got', len(self.disambig_page_ids), 'disambiguation page ids'
        else:
            self.disambig_page_ids = set()
            disambig_page_names = ttl.read_ttl_keys(path)
            for i, page in enumerate(disambig_page_names):
                if i % 1000 == 0:
                    print 'resolving disambig pages,', i
                page_id = db.resolvePage(page)
                if page_id is not None:
                    self.disambig_page_ids.add(page_id)
            print 'got', len(self.disambig_page_ids), 'disambiguation page ids'
            pickle.dump(self.disambig_page_ids, open(path + '.id_cache', 'w'))

    def filter_candidates(self, candidates):
        return {cand for cand in candidates if cand not in self.disambig_page_ids}


if __name__ == "__main__":
    _path = "/home/yotam/pythonWorkspace/deepProject"
    from DbWrapper import *
    from DbWrapperCached import *

    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
    # wikiDB = WikipediaDbWrapper(user='noambox', password='ncTech#1', database='wiki20151002')
    yago2 = CandidatesUsingYago2(None)
    yago2.import_yago("../../data/yago/aida_means.tsv", wikiDB)
    yago2.save("../../data/yago/yago.candidates.real")
