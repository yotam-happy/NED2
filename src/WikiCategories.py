from utils.read_dbpedia_ttl import *
import pickle
import os
import math

class WikiCategories:
    def __init__(self, embd_sz):
        self.category_id2title = dict()
        self.entity_categories = dict()
        self.embeddings = None
        self.embd_sz = embd_sz

    def get_number_of_values(self):
        return len(self.category_id2title)

    def get_embd_sz(self):
        return self.embd_sz

    def get_empty_id(self):
        return 0

    def get_dummy_id(self):
        return 1

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
        self.cat_embds = dict()
        self.cat_sz = dict()
        self.cat_embd_dict = dict()
        self.zzz = None

    def get_categories_for_page(self, page_id, max_depth=3, max_accepted_depth=1, w2v=None, useCache=True):
        if useCache and page_id in self.page_categories_cache:
            return self.page_categories_cache[page_id]
        if page_id not in self.categorylinks:
            return set()

        if w2v is not None and page_id not in w2v.conceptDict:
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

        '''
        if w2v is not None:
            page_embd = w2v.get_entity_vec(page_id)
            cat_sim = dict()
            for cat in self.page_categories_cache[page_id]:
                cat_embd = self.cat_embds[cat]
                cat_sz = self.cat_sz[cat]

                if self.zzz is not None:
                    print "z", w2v.similarity(self.zzz, page_embd)
                self.zzz = page_embd
                cat_sim[cat] = w2v.similarity(cat_embd, page_embd)

            if len({x for x in cat_sim.itervalues() if x < 1.0}) == 0:
                return cat_sim

            max_sim = max(cat_sim.itervalues())
            min_sim = min(cat_sim.itervalues())
            if max_sim - min_sim > 0:
                for cat in cat_sim.keys():
                    cat_sim[cat] = (cat_sim[cat] - min_sim) / max_sim

            self.page_categories_cache[page_id] = [x for x in self.page_categories_cache[page_id]
                                                   if x in cat_sim and cat_sim[x] < 1]
            print 'categories for', wikiDB.getPageTitle(page_id), ':', {self.category_id2title[x]:cat_sim[x] for x in self.page_categories_cache[page_id]}
        '''

        return {x for x in self.page_categories_cache[page_id]}

    def saveCategoryData(self, page_ids, w2v, temp_file='categories_embeds.temp.pickle'):
        print 'processing category embds. Going over all pages'
        embds_dict = dict()
        self.cat_sz = dict()
        for i, page_id in enumerate(page_ids):
            page_vec = w2v.get_entity_vec(page_id)
            if page_vec is None:
                continue

            # get all categories this page belongs to - no filters
            cats = self.get_categories_for_page(page_id)
            for cat in cats:
                if cat not in embds_dict:
                    embds_dict[cat] = np.copy(page_vec)
                    self.cat_sz[cat] = 1
                else:
                    embds_dict[cat] += page_vec
                    self.cat_sz[cat] += 1

            if i % 1000 == 0:
                print 'done', i

        print 'normalize...'
        for cat_name, sz in self.cat_sz.iteritems():
            embds_dict[cat_name] /= sz

        # make a single large numpy matrix. the zero entry is kept for -no category- marker
        self.cat_embds = np.zeros((len(embds_dict) + 2, w2v.conceptEmbeddingsSz))
        self.cat_embd_dict = dict()
        self.cat_embd_dict['~Empty~'] = 0
        self.cat_embd_dict['~Dummy~'] = 1
        i = 2
        for cat, embd in embds_dict.iteritems():
            self.cat_embds[i, :] = embd
            self.cat_embd_dict[cat] = i
            i += 1

        # calc page to category
        page_categories = dict()
        page_categories['~Empty~'] = 0
        page_categories['~Dummy~'] = 1

        for i, page_id in enumerate(page_ids):
            page_vec = w2v.get_entity_vec(page_id)
            if page_vec is None:
                continue

            # get all categories this page belongs to - no filters
            cats = self.get_categories_for_page(page_id)
            page_categories[page_id] = {self.cat_embd_dict[cat] for cat in cats}
            print page_id, {self.cat_embd_dict[cat] for cat in cats}

        print 'cat embds:', self.cat_embds.shape

        np.save(temp_file + '.npy', self.cat_embds)
        with open(temp_file, 'w') as f:
            pickle.dump(page_categories, f)
            pickle.dump(w2v.conceptEmbeddingsSz, f)

    def getWikiCategories(self, page_ids, w2v=None):
        id_convert = dict()
        wiki_cats = WikiCategories(None)
        wiki_cats.category_id2title[0] = '~Empty~'
        wiki_cats.category_id2title[1] = '~Dummy~'
        for i, page_id in enumerate(page_ids):
            cats = self.get_categories_for_page(page_id, w2v=w2v)
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

    w2v = Word2vecLoader(wordsFilePath="/home/yotam/pythonWorkspace/deepProject/data/word2vec/dim300vecs",
                         conceptsFilePath="/home/yotam/pythonWorkspace/deepProject/data/word2vec/dim300context_vecs")
    concept_filter = {int(x) for x in stats.conceptCounts}
    w2v.loadEmbeddings(conceptDict=concept_filter)

    wikiCats = WikiCategoriesBuilder()
    wikiCats.load('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/article_categories_en.ttl.cache')
    wikiCats.process_topical_categories('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/topical_concepts_en.ttl', wikiDB)

    print 'getting and saving categories for', len(concept_filter), 'pages'
    cc = wikiCats.getWikiCategories(concept_filter)
    cc.save('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/wiki_categories.dump')
    wikiCats.saveCategoryData(concept_filter, w2v, '/home/yotam/pythonWorkspace/deepProject/data/word2vec/category_vecs.temp')

    print 'done'