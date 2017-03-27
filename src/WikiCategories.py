from utils.read_dbpedia_ttl import *
import pickle
import os
import math

class WikiCategories:
    def __init__(self, embd_sz):
        self.category_id2title = dict()
        self.entity_categories = dict()
        self.embd_sz = embd_sz

    def get_number_of_values(self):
        return len(self.category_id2title)

    def get_embd_sz(self):
        return self.embd_sz

    def get_empty_id(self):
        return 0

    def get_categories(self, entity_id):
        return self.entity_categories[entity_id] if entity_id in self.entity_categories else []

    def save(self, f):
        # save results
        with open(f, 'w') as f:
            pickle.dump(self.category_id2title, f)
            pickle.dump(self.entity_categories, f)


    def load(self, f):
        with open(f, 'r') as f:
            self.category_id2title = pickle.load(f)
            self.entity_categories = pickle.load(f)


class WikiCategoriesBuilder:
    '''
    self.categorylinks = from -> to
    self.categorylinks_inv = to -> from
    self.sub_cats = parent -> child
    self.sub_cats_inv = child -> parent
    '''

    def __init__(self):
        self.category_id2title = dict()
        self.category_title2id = dict()
        self.sub_cats = dict()
        self.parent_cats = dict()
        self.categorylinks = dict()
        self.page_categories_cache = dict()
        self.topical_categories = set()

    def calc_category_infogain(self, training_iter, candidator, db, w2v=None):
        # E(T) = entropy for correct senses out of population
        # E(T|rejected_by_category_i) = expected entropy of correct senses out of population given category i
        # notice we condition on the event rejected_by_category_i so one group would be all the candidates
        # that are immediately rejected having know category_i and the second group is the correct candidates +
        # candidates that are not immdeiately rejected having known category_i

        total_cases = 0
        total_candidates = 0

        # keeps how many candidates would be eliminated having known a category
        category_rejection_count = dict()
        category_candidate_sz = dict()

        print "calculating info gain..."
        for j, mention in enumerate(training_iter.mentions()):
            candidates = candidator.get_candidates_for_mention(mention)
            total_cases += 1
            total_candidates += len(candidates)

            # collect categories for candidates
            candidate_categories = self.get_categories_for_pages(candidates, db, w2v=w2v)
            for category, pages_in_category in candidate_categories.iteritems():
                category_candidate_sz[category] = category_candidate_sz.get(category, 0) + \
                                                     len(pages_in_category)
                if mention.gold_sense_id() in pages_in_category:
                    # can eliminated pages not in this category
                    category_rejection_count[category] = category_rejection_count.get(category, 0) + \
                                                         len(candidates) - len(pages_in_category)
                else:
                    # can eliminate all pages in this category
                    category_rejection_count[category] = category_rejection_count.get(category, 0) + \
                                                         len(pages_in_category)
            if j == 3:
                z = 0 / 0

            if j % 1000 == 0:
                print 'done', j, 'mentions'

        avg_candidates_per_case = float(total_cases) / total_candidates
        dataset_entropy = - avg_candidates_per_case * math.log(avg_candidates_per_case, 2) \
                          - (1-avg_candidates_per_case) * math.log((1-avg_candidates_per_case), 2)

        category_infogain = dict()
        for category, rejection_count in category_rejection_count.iteritems():
            p_rejected = float(rejection_count) / total_candidates
            avg_candidates_per_case_after_rejection = float(total_candidates - rejection_count) / total_cases
            frac_correct_after_rejection = 1 / float(avg_candidates_per_case_after_rejection)
            conditional_entropy = - frac_correct_after_rejection * math.log(frac_correct_after_rejection, 2) \
                                  - (1-frac_correct_after_rejection) * math.log((1-frac_correct_after_rejection), 2)
            # the entropy of the rejected group is 0
            expected_entropy = (1-p_rejected) * conditional_entropy
            category_infogain[category] = dataset_entropy - expected_entropy
        return category_infogain

    def get_categories_for_page(self, page_id, db, max_depth=3, max_accepted_depth=1, w2v=None):
        if page_id in self.page_categories_cache:
            return self.page_categories_cache[page_id]
        if page_id not in self.categorylinks:
            return set()
        #if w2v is not None and page_id not in w2v.conceptDict:
        #    return set()
        if w2v is not None:
            embd1 = w2v.into_the_cache(page_id, 'asdasdg', 'afgadfgdf')
            if embd1 is None:
                return set()

        # get all categories for the page (using BFS)
        processed = set()
        front = {category for category in self.categorylinks[page_id]}
        depth = {category: 0 for category in front}
        accepted_depth = {category: 0 for category in front}
        child = {category: None for category in front}
        steps = 0
        while len(front) > 0:
            new_front = set()
            for category in front:
                d = depth[category] + 1
                ad = accepted_depth[category] + (1 if self.category_filter(category) else 0)
                if d < max_depth and ad < max_accepted_depth:
                    for x in self.parent_cats[category]:
                        if x not in processed and x not in front:
                            new_front.add(x)
                            child[x] = category if x not in depth or depth[x] > d else child[x]
                            depth[x] = d if x not in depth or depth[x] > d else depth[x]
                            accepted_depth[x] = ad if x not in accepted_depth or accepted_depth[x] > ad else accepted_depth[x]
            processed.update(front)
            front = new_front
            steps += 1
        self.page_categories_cache[page_id] = [x for x in processed
                                               if self.category_filter(x) and depth[x] < max_depth
                                               and accepted_depth[x] < max_accepted_depth]

        cat_dist = dict()
        for cat in self.page_categories_cache[page_id]:
            cat_page = self.topical_categories[cat]
            if w2v is not None:
                #                embd1 = w2v.conceptEmbeddings[w2v.conceptDict[page_id],:]
                #                embd2 = w2v.conceptEmbeddings[w2v.conceptDict[cat_page],:]
                embd2 = w2v.into_the_cache(cat_page, 'asfgadgd', 'adsfgdfadgdag')
                if embd2 is not None:
                    cat_dist[cat] = w2v.similarity(embd1, embd2)
        #mean_dist = float(reduce(lambda x, y: x + y, cat_dist.itervalues())) / len(cat_dist)
        #print 'mean', mean_dist, cat_dist
        #self.page_categories_cache[page_id] = [x for x in self.page_categories_cache[page_id]
        #                                       if cat_dist[x] > mean_dist / 2]
        if len({x for x in cat_dist.itervalues() if x < 1.0}) == 0:
            return cat_dist
        max_dist = max({x for x in cat_dist.itervalues() if x < 1.0})
        #print 'max', max_dist, cat_dist
        self.page_categories_cache[page_id] = [x for x in self.page_categories_cache[page_id]
                                               if x in cat_dist and cat_dist[x] > max_dist / 2]


        #page_name = db.getPageTitle(page_id)
        #print 'for page', page_name
        #for cat in self.page_categories_cache[page_id]:
        #    dist = cat_dist[cat] if cat in cat_dist else 'N/A'
        #    path = '->' + self.category_id2title[cat]
        #    c = child[cat]
        #    while c is not None:
        #        path = '->' + self.category_id2title[c] + path
        #        c = child[c]
        #    print '     dist:', dist, 'path:', path
        #print '  -- categories:', {self.category_id2title[x] for x in self.page_categories_cache[page_id]}

        return {x for x in self.page_categories_cache[page_id]}

    def get_categories_for_pages(self, page_ids, db, w2v=None):
        categories_for_pages = dict()
        for page_id in page_ids:
            cats = self.get_categories_for_page(page_id, db, w2v=w2v)
            # add to categories dict
            for category in cats:
                if category not in categories_for_pages:
                    categories_for_pages[category] = set()
                categories_for_pages[category].add(page_id)

        return categories_for_pages

    def getWikiCategories(self, page_ids, db, w2v=None):
        id_convert = dict()
        wiki_cats = WikiCategories(None)
        wiki_cats.category_id2title[0] = '~Empty~'
        for i, page_id in enumerate(page_ids):
            cats = self.get_categories_for_page(page_id, db, w2v=w2v)
            for cat in cats:
                if cat not in id_convert:
                    id_convert[cat] = len(id_convert)
                    wiki_cats.category_id2title[id_convert[cat]] = self.category_id2title[cat]
            wiki_cats.entity_categories[page_id] = {id_convert[x] for x in cats}
            if i % 1000 == 0:
                print 'done', i
        print 'got', len(wiki_cats.category_id2title) - 1, 'categories'
        return wiki_cats

    def process_sub_categories(self, db):
        self.sub_cats = {cat_id: set() for cat_id in self.category_id2title.iterkeys()}
        self.parent_cats = {cat_id: set() for cat_id in self.category_id2title.iterkeys()}

        # Recursively resolve sets of parent/child categories
        query = "select cl_from, cl_to from categorylinks where cl_type=%s"
        cnx = db.getConnection(timeout=100)
        fetch_cursor = cnx.cursor(buffered=False)
        fetch_cursor.execute(query, ("subcat",))
        i = 0
        j = 0
        while True:
            row = fetch_cursor.fetchone()
            if not row:
                break
            if str(row[1]) in self.category_title2id \
                    and int(row[0]) in self.category_id2title:
                j += 1
                child = int(row[0])
                parent = self.category_title2id[str(row[1])]
                self.sub_cats[parent].add(child)
                self.parent_cats[child].add(parent)
            i += 1
            if i % 100000 == 0:
                print 'processed', i, 'rows. got', j, 'subcat links.', float(j) / i

    def process_page_category_links(self, db):
        # get parent categories for each concept
        self.categorylinks = dict()
        query = "select cl_from, cl_to from categorylinks where cl_type=%s"
        cnx = db.getConnection(timeout=100)
        fetch_cursor = cnx.cursor(buffered=False)
        fetch_cursor.execute(query, ("page",))
        i = 0
        j = 0
        while True:
            row = fetch_cursor.fetchone()
            if not row:
                break
            page_id = db.resolvePage(int(row[0]))

            if str(row[1]) in self.category_title2id \
                    and page_id is not None:
                j += 1
                category = self.category_title2id[str(row[1])]
                if page_id not in self.categorylinks:
                    self.categorylinks[page_id] = set()
                self.categorylinks[page_id].add(category)
            i += 1
            if i % 100000 == 0:
                print 'processed', i, 'rows. got', j, 'page-category links.', float(j) / i

    def process_page_category_links_for_entity(self, db):
        # get parent categories for each concept
        self.categorylinks = dict()
        query = "select cl_from, cl_to from categorylinks where cl_type=%s"
        cnx = db.getConnection(timeout=100)
        fetch_cursor = cnx.cursor(buffered=False)
        fetch_cursor.execute(query, ("page",))
        i = 0
        j = 0
        while True:
            row = fetch_cursor.fetchone()
            if not row:
                break
            page_id = db.resolvePage(int(row[0]))

            if str(row[1]) in self.category_title2id \
                    and page_id is not None:
                j += 1
                category = self.category_title2id[str(row[1])]
                if page_id not in self.categorylinks:
                    self.categorylinks[page_id] = set()
                self.categorylinks[page_id].add(category)
            i += 1
            if i % 100000 == 0:
                print 'processed', i, 'rows. got', j, 'page-category links.', float(j) / i

    def process_db(self, db):
        if os.path.exists('categories.temp.pickle'):
            with open('categories.temp.pickle', 'r') as f:
                self.category_id2title = pickle.load(f)
        else:
            self.category_id2title = db.getAllCategories()
            with open('categories.temp.pickle', 'w') as f:
                pickle.dump(self.category_id2title, f)

        self.category_title2id = {y: x for x, y in self.category_id2title.iteritems()}

        if os.path.exists('subcategories.temp.pickle'):
            with open('subcategories.temp.pickle', 'r') as f:
                self.sub_cats = pickle.load(f)
                self.parent_cats = pickle.load(f)
        else:
            self.process_sub_categories(db)
            with open('subcategories.temp.pickle', 'w') as f:
                pickle.dump(self.sub_cats, f)
                pickle.dump(self.parent_cats, f)

        self.process_page_category_links(db)

    def process_topical_categories(self, ttl_path, db):
        '''
        returns list of categories described by a topical concept (we take these to be
        categories that describe actual abstract entities e.g. People, Libraries, Desserts...)
        :param ttl_path: path to topical_concepts_en.ttl from DbPedia
        :return:
        '''
        topical_categories = dict()
        for i, (cat, page) in enumerate(read_ttl_generator(ttl_path)):
            if not cat.startswith('Category:') or cat[9:] not in self.category_title2id:
                continue
            page_id = db.resolvePage(page)
            if page_id is None:
                continue
            topical_categories[self.category_title2id[cat[9:]]] = page_id
        print "num of topical categories", len(topical_categories)
        self.topical_categories = topical_categories

        '''
    def process_ttl_files(self, article_categories_ttl, skos_categories_ttl, db):

        # NOTE: no sub categories!
        self.category_page = dict()
        self.page_category = dict()

        k = 0
        for i, (page, cat) in enumerate(read_ttl_generator(article_categories_ttl)):
            page_id = db.resolvePage(page)
            cat_id = db.getCategoryByName(cat[9:])
            if i % 1000 == 0:
                print "got", k, 'category links out of', i
            if page_id is None:
                continue
            if cat_id is None:
                continue
            k += 1
            if page_id not in self.page_category:
                self.page_category[page_id] = set()
            if cat_id not in self.category_page:
                self.category_page[cat_id] = set()
            self.page_category[page_id].add(cat_id)
            self.category_page[cat_id].add(page_id)
    '''

    def save(self, f):
        # save results
        with open(f, 'w') as f:
            pickle.dump(self.category_id2title, f)
            pickle.dump(self.category_title2id, f)
            pickle.dump(self.sub_cats, f)
            pickle.dump(self.parent_cats, f)
            pickle.dump(self.categorylinks, f)

    def load(self, f):
        with open(f, 'r') as f:
            self.category_id2title = pickle.load(f)
            self.category_title2id = pickle.load(f)
            self.sub_cats = pickle.load(f)
            self.parent_cats = pickle.load(f)
            self.categorylinks = pickle.load(f)

    def filter_categories(self, cats):
        return {x for x in cats if self.category_filter(x)}

    def category_filter(self, cat_id):
        title = self.category_id2title[cat_id]
        if '_with_' in title or \
                '_by_' in title or \
                '_in_' in title or \
                '_identifiers' in title or \
                cat_id not in self.topical_categories:
            return False
        return True

if __name__ == "__main__":
    from DbWrapper import *
    from WikilinksIterator import *
    from Candidates import *
    from WikilinksStatistics import *
    from Word2vecLoader import *
    from yamada.text_to_embedding import *

    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
    stats = WikilinksStatistics(None, load_from_file_path=
                                '/home/yotam/pythonWorkspace/deepProject/data/wikilinks/filtered-train-stats.final')

#    w2v = Word2vecLoader(wordsFilePath="/home/yotam/pythonWorkspace/deepProject/data/word2vec/dim300vecs",
#                         conceptsFilePath="/home/yotam/pythonWorkspace/deepProject/data/word2vec/dim300context_vecs")
#    concept_filter = {int(x) for x in stats.conceptCounts}
#    w2v.loadEmbeddings(conceptDict=concept_filter)
    w2v = YamadaEmbedder("/home/yotam/pythonWorkspace/deepProject/data/yamada/enwiki_entity_vector_500_20151026.pickle",
                         "/home/yotam/pythonWorkspace/deepProject/data/yamada/id2title.tsv",
                         db=wikiDB)

    wikiCats = WikiCategoriesBuilder()
    #wikiCats.process_db(wikiDB)
    #wikiCats.save('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/article_categories_en.ttl.cache')
    wikiCats.load('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/article_categories_en.ttl.cache')
    wikiCats.process_topical_categories('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/topical_concepts_en.ttl', wikiDB)

    print 'getting and saving categories for pages'
    cc = wikiCats.getWikiCategories(stats.conceptCounts, wikiDB, w2v)
    cc.save('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/wiki_categories.dump')
    print 'done'




    #n = 5000
    #training_iter = WikilinksNewIterator('/home/yotam/pythonWorkspace/deepProject/data/wikilinks/filtered/train')
    #candidator = CandidatesUsingStatisticsObject(stats, db=wikiDB)
    #category_infogain = wikiCats.calc_category_infogain(training_iter, candidator, wikiDB, w2v=w2v)
    #with open('tmp.tmp.tmp', 'w') as f:
    #    pickle.dump(category_infogain, f)

    #with open('tmp.tmp.tmp', 'r') as f:
    #    category_infogain = pickle.load(f)


    #category_infogain_sorted = [x for x in category_infogain.iterkeys()]
    #category_infogain_sorted.sort(key=lambda k: -category_infogain[k])
    #print "some best categories:"
    #for i,(cat) in enumerate(category_infogain_sorted):
    #    print wikiCats.category_id2title[cat], ':', category_infogain[cat]
    #    if i > 50:
    #        break
    #best_n = category_infogain_sorted #{x for x in category_infogain_sorted[:n]}

    #cat_pages = {x: 0 for x in best_n}
    #page_categories = dict()
    #print 'getting category list for each page:'
    #hist = [0 for x in xrange(20)]
    #for i, page_id in enumerate(stats.conceptCounts):
    #    cats = {x for x in wikiCats.get_categories_for_page(page_id)}
    #    #for cat in cats:
    #    #    cat_pages[cat] += 1
    #    hist[len(cats) if len(cats) < 19 else 19] += 1
    #    if i % 100 == 0:
    #        print 'done', i, 'pages'
    #print 'hist of categories per page', hist
    #print 'using', len(best_n), 'categories out of', len(category_infogain_sorted)

    #hist_cats = [0 for x in xrange(20)]
    #for cat, pages in cat_pages.iteritems():
    #    hist_cats[pages if pages < 19 else 19] += 1
    #print 'hist of pages per category', hist_cats

    #wikiCats.process_ttl_files('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/article_categories_en.ttl',
    #                           '/home/yotam/pythonWorkspace/deepProject/data/DbPedia/skos_categories_en.ttl',
    #                           wikiDB)
    #print len(wikiCats.page_category), 'page is', len(wikiCats.category_page), 'categories'

    #wikiCats.load('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/article_categories_en.ttl.cache')
