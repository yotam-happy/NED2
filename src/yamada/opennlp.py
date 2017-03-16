from jsonrpclib.jsonrpc import ServerProxy
import nltk


def _parse_tree(parse_str):
    """
    returns a 2-tuple. first item is the tag, second is either a word or a list representing a sub-tree

    :param self:
    :param parse_str:
    :return:
    """
    if parse_str[0] != '(':
        raise "error, parse string malformed!"
    rest = parse_str[1:]

    i = rest.find(' ')
    if i < 1:
        raise "error, parse string malformed!"
    tag = rest[:i]
    rest = rest[i:]

    # if we reached a leaf.
    if rest[1] != '(':
        rest = rest[1:]
        i = rest.find(')')
        k = rest.find('(')
        if k >=0 and k < i:
            raise "error, parse string malformed!"
        node = rest[:i]
        rest = rest[i:]
        return (tag, node), rest

    node_list = []
    while rest.startswith(' ('):
        rest = rest[1:]
        subtree, rest = _parse_tree(rest)
        node_list.append(subtree)
        if rest[0] != ')':
            raise "error, parse string malformed!"
        rest = rest[1:]
    tree = (tag, node_list)
    if len(rest) == 1:
        return tree
    else:
        return tree, rest


def print_tree(tree, sp=''):
    s = sp
    tag, node = tree
    s += tag
    if type(node) is not list:
        s += ' ' + str(node)
        print s
    else:
        print s
        for subree in node:
            print_tree(subree, sp + "  ")


def _list_nouns(tree, l):
    tag, node = tree
    if type(node) is not list:
        if tag.startswith('N'):
            l.append(node)
    else:
        for subree in node:
            _list_nouns(subree, l)


class OpenNLP:
    def __init__(self, host='localhost', port=8080):
        uri = "http://%s:%d" % (host, port)
        self.server = ServerProxy(uri)

    def parse(self, text):
        p = None
        try:
            p = self.server.parse(text)
            return _parse_tree(p)
        except:
            print "problem with parse tree:", p
            return None

    def _list_nouns(self, sentences):
        nouns = []
        for s in sentences:
            l = []
            p = self.parse(s)
            if p is not None:
                try:
                    _list_nouns(p, l)
                    nouns += l
                except:
                    pass
        return nouns

    def list_nouns(self, sentences):
        nouns = []
        for s in sentences:

            tagged = nltk.pos_tag(s.split(' '))
            for word, tag in tagged:
                if tag.startswith('N'):
                    nouns.append(word)
        return nouns


if __name__ == '__main__':
    nlp = OpenNLP()
    results = nlp.parse("Shhh  (Be vewy vewy quiet), I'm hunting wabbits .")
    print_tree(results)
    print nlp.list_nouns("Shhh  (Be vewy vewy quiet), I'm hunting wabbits .")