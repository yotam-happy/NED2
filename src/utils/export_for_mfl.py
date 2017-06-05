from DbWrapper import *
from WikilinksIterator import *
from spacy.en import English


def do_for_subfolder(subfolder):
    # file type sould be conll 2012 format. tab delimitered with one row for each word. empty row for sentence bound:
    # 1. doc id
    # 2. part no (can be always 0)
    # 3. word num
    # 4. word
    # 5. pos tag
    # 6. parse tree until current word. the word as *
    # 7. always -
    # 8. always -
    # 9. always -
    # 10. always -
    # 11. type of named entity with span
    # 12. coref

    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')

    _path = "/home/yotam/pythonWorkspace/deepProject"

    parser = English()

    outdir = _path + "/data/wikilinks/for_MFL_step1/" + subfolder
    it = WikilinksNewIterator(_path+"/data/wikilinks/filtered/" + subfolder)
    split = subfolder.upper()

    for i, wlink in enumerate(it.jsons()):
        if i % 100 == 0:
            print subfolder, ': done', i

        # if we start from an interrupted job - redo the last 1000 files in case some dist writing was interrupted
        if os.path.exists(os.path.join(outdir, 'WIKILINKS.' + split + '.' + str(i+1000))):
            continue
        txt = wlink['left_context_text'] + ' ' + wlink['word'] + ' ' + wlink['right_context_text']

        mention_sent_start_index = 0
        mention_sent_index = 0
        mention_start_idx = len(wlink['left_context_text'] + ' ')
        mention_end_idx = len(wlink['left_context_text'] + ' ' + wlink['word']) - 1
        gold_wiki_title = wikiDB.getPageTitle(wlink['wikiId']).replace('(', '-LRB-').replace(')', '-RRB-')

        parsedData = parser(txt)
        sents = []
        for k, sent in enumerate(parsedData.sents):
            if sent.start_char <= mention_start_idx:
                mention_sent_start_index = sent.start_char
                mention_sent_index = k

            #sents.append(sent)
        with open(os.path.join(outdir, 'WIKILINKS.' + split + '.' + str(i)), 'w+') as f:
            f.write(str(mention_sent_index) + '\t' +
                    str(mention_start_idx - mention_sent_start_index) + '\t' +
                    str(mention_end_idx - mention_sent_start_index) + '\t' +
                    gold_wiki_title + '\n')
            for sent in parsedData.sents:
                f.write(str(sent) + '\n')

    print 'done'

if __name__ == "__main__":
    #do_for_subfolder('test')
    do_for_subfolder('train')
