from utils.read_dbpedia_ttl import *
import pickle

class WikiCategories2:
    def __init__(self):
        self.category_page = dict()
        self.page_category = dict()

class WikiCategories:
    '''
    Handle dbpedia's article_categories_en.ttl and skos_categories_en.ttl
    resolves all page and category redirects
    keeps as dictionaries
    '''

    def __init__(self):
        self.category_page = dict()
        self.page_category = dict()

    def process_ttl_files(self, article_categories_ttl, skos_categories_ttl, db):
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

    def save(self, f):
        # save results
        with open(f, 'w') as f:
            pickle.dump(self.category_page, f)
            pickle.dump(self.page_category, f)

    def load(self, f):
        with open(f, 'r') as f:
            self.category_page = pickle.load(f)
            self.page_category = pickle.load(f)

if __name__ == "__main__":
    from DbWrapper import *
    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
    wikiCats = WikiCategories()
    #wikiCats.process_ttl_files('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/article_categories_en.ttl',
    #                           '/home/yotam/pythonWorkspace/deepProject/data/DbPedia/skos_categories_en.ttl',
    #                           wikiDB)
    #wikiCats.save('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/article_categories_en.ttl.cache')
    #print len(wikiCats.page_category), 'page is', len(wikiCats.category_page), 'categories'

    wikiCats.load('/home/yotam/pythonWorkspace/deepProject/data/DbPedia/article_categories_en.ttl.cache')
    cats = [x for x in wikiCats.category_page.iterkeys()]
    cats.sort(key=lambda x: -len(wikiCats.category_page[x]))
    for i, cat in enumerate(cats):
        print cat, wikiDB.getCategoryTitle(cat), len(wikiCats.category_page[cat])
        if i > 100:
            break

    print len(wikiCats.category_page[wikiDB.getCategoryByName('Disambiguation_pages')])
