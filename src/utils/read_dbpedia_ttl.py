import re

def read_ttl(path):
    print 'reading ttl:', path
    mapping = dict()
    sz = 0
    with open(path, 'r') as f:
        for l in f:
            triplet = re.findall("<[^<>]+>", l)
            if len(triplet) != 3:
                continue
            k = triplet[0][1: -1]
            k = k[k.rfind('/')+1:]
            v = triplet[2][1: -1]
            v = v[v.rfind('/')+1:]
            if k not in mapping:
                mapping[k] = set()
            mapping[k].add(v)
            sz += 1
    print 'got', len(mapping), 'keys with', sz, 'items from', path
    return mapping

def read_ttl_generator(path):
    print 'reading ttl as generator:', path
    with open(path, 'r') as f:
        for l in f:
            triplet = re.findall("<[^<>]+>", l)
            if len(triplet) != 3:
                continue
            k = triplet[0][1: -1]
            k = k[k.rfind('/')+1:]
            v = triplet[2][1: -1]
            v = v[v.rfind('/')+1:]
            yield k, v

def read_ttl_keys(path):
    print 'reading ttl:', path
    keys = set()
    with open(path, 'r') as f:
        for l in f:
            triplet = re.findall("<[^<>]+>", l)
            if len(triplet) != 3:
                continue
            k = triplet[0][1: -1]
            k = k[k.rfind('/')+1:]
            v = triplet[2][1: -1]
            v = v[v.rfind('/')+1:]
            keys.add(k)
    print 'got', len(keys), 'keys from', path
    return keys

if __name__ == "__main__":
    read_ttl('../../../data/DbPedia/disambiguations_en.ttl')
