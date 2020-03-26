import time
from gensim.models import Word2Vec
from deepwalkalgo.walker import BasicWalker

class DeepWalk(object):
    def __init__(self, graph, path_length, num_paths, dim, **kwargs):
        kwargs["workers"] = kwargs.get("workers", 1)
        kwargs["hs"] = 1
        self.graph = graph
        self.walker = BasicWalker(graph, workers=kwargs["workers"])
        sentences = self.walker.simulate_walks(num_walks=num_paths, walk_length=path_length)
        kwargs["sentences"] = sentences
        kwargs["min_count"]  = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size",dim)
        kwargs["sg"] = 1
        
        self.size = kwargs["size"]
        print("learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        for word in graph.G.nodes():
            self.vectors[word] = word2vec.wv[word]
        del word2vec


    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()