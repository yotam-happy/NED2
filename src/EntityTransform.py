from utils.read_dbpedia_ttl import *
import os
import pickle

class EntityToTypeByDbPediaLhdDataset:
    def __init__(self, ttl_path, db, embd_sz):
        self.embd_sz = embd_sz

        if os.path.isfile(ttl_path + '.cached'):
            with open(ttl_path + '.cached', 'r') as f:
                self.entity_types = pickle.load(f)
                self.typeid_to_str = pickle.load(f)
            print 'got', len(self.entity_types), 'entity type definitions with', len(self.typeid_to_str), 'types'
            return

        entity_types_raw = read_ttl(ttl_path)

        # enumerate types
        type_counts = get_value_marginals(entity_types_raw)
        enum_types = dict()
        self.typeid_to_str = dict()
        self.typeid_to_str[0] = 'NO_TYPE'
        for i, t in enumerate(type_counts.iterkeys()):
            enum_types[t] = i + 1
            self.typeid_to_str[i + 1] = t

        # for a small number of entities, there's more then one type. We use a simple
        # and by no means optimal heuristic - we pick the more common type.
        self.entity_types = dict()
        i = 0
        print 'resolving entity types'
        for k, v in entity_types_raw.iteritems():
            ent_type = max(v, key=lambda x: type_counts[x])
            ent_type = enum_types[ent_type]
            ent_id = db.resolvePage(k)
            if ent_id is not None:
                self.entity_types[ent_id] = ent_type
            i += 1
            if i % 1000 == 0:
                print 'done', i, 'got', len(self.entity_types)

        # save results
        with open(ttl_path + '.cached', 'w') as f:
            pickle.dump(self.entity_types, f)
            pickle.dump(self.typeid_to_str, f)
        print 'got', len(self.entity_types), 'entity type definitions with', len(self.typeid_to_str), 'types'

    def entity_id_transform(self, entity_id):
        return self.entity_types[entity_id] if entity_id in self.entity_types else 0

    def id2string(self, eid):
        return self.typeid_to_str[eid] if eid in self.typeid_to_str else '<not found>'

    def get_number_of_values(self):
        return len(self.typeid_to_str)

    def get_embd_sz(self):
        return self.embd_sz


def get_value_marginals(map_of_sets):
    value_marginals = dict()
    for v_set in map_of_sets.itervalues():
        for v in v_set:
            value_marginals[v] = 1 + value_marginals.get(v, 0)
    return value_marginals

if __name__ == "__main__":
    from DbWrapper import *
    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
    transform = EntityToTypeByDbPediaLhdDataset('../../data/DbPedia/instance_types_lhd_dbo_en.ttl', wikiDB)
    print transform.entity_types
