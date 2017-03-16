import nltk
from entity_vector import EntityVector
import numpy as np
from yamada.opennlp import *
import math
import codecs

class YamadaEmbedder:
    def __init__(self, path, id2title_path, db=None):
        self.id2title = dict()
        with codecs.open(id2title_path, 'r', 'utf-8') as f:
            for line in iter(f):
                l = line.split('\t')
                self.id2title[int(l[0])] = l[1]
        print 'title convertion table for yamada:', len(self.id2title)
        self.parser = OpenNLP()
        self.entvec = EntityVector.load(path)
        self._cache = dict()
        self._db = db
        self.n = 0
        self.nn = 0

        self.nnn = 0
        self.nnnn = 0

        return

    def similarity(self, v1, v2):
        return np.dot(v1, v2) if v1 is not None and v2 is not None else 0

    def text_to_embedding(self, nouns, mention):
        vecs = []
        for word in nouns:
            if not mention.lower().find(word.lower()) > -1:
                try:
                    self.nnn += 1
                    w = self.entvec.get_word_vector(unicode(word.lower()))
                    if w is not None:
                        vecs.append(w)
                        self.nnnn += 1
                    if self.nnn % 100 == 0:
                        print "context resolve", float(self.nnnn) / self.nnn
                except:
                    pass
        return np.array(vecs).mean(0) if len(vecs) > 0 else None

    def entity_embd(self, title):
        try:
            return self.entvec.get_entity_vector(title)
        except:
            return None

    def from_the_cache(self, page_id):
        return self._cache[page_id] if page_id in self._cache else None

    def into_the_cache(self, page_id, url, title, verbose=False):
        if self.n > 0 and self.n % 100 == 0:
            print "not resolved ", float(self.nn) / self.n, "out of", self.n
        if page_id in self._cache:
            self.n += 1
            return self._cache[page_id]
        self.n += 1
        try:
            if page_id in self.id2title:
                embd = self.entity_embd(self.id2title[page_id])
                if embd is not None:
                    self._cache[page_id] = embd
                    return embd

            if title is not None:
                title = title.replace('_', ' ')
                embd = self.entity_embd(title.decode("utf-8"))
                if embd is not None:
                    self._cache[page_id] = embd
                    return embd

            title_from_url = unicode(url.decode("utf-8"))
            title_from_url = title_from_url[title_from_url.rfind('/') + 1:]
            title_from_url = title_from_url.replace('_', ' ')
            embd = self.entity_embd(title_from_url)
            if embd is not None:
                self._cache[page_id] = embd
                return embd

            url_by_id = self._db.getPageTitle(page_id)
            url_by_id = unicode(url_by_id.decode("utf-8"))
            url_by_id = url_by_id.replace('_', ' ')
            embd = self.entity_embd(url_by_id)
            if embd is not None:
                self._cache[page_id] = embd
                return embd
        except Exception as e:
            self.nn += 1
            if verbose:
                print url, "some error", e
            return None
        self.nn += 1
        if verbose:
            print url, " not resolved"
        return None


if __name__ == "__main__":
   embedder = YamadaEmbedder('../data/yamada/enwiki_entity_vector_500_20151026.pickle')
   txt = "The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep ."
   print embedder.text_to_embedding(txt)