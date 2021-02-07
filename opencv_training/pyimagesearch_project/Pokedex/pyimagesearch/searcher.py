from scipy.spatial import distance as dis

class Searcher:
    def __init__(self, index):
        self.index = index
    
    def search(self, queryFeatures):

        results = {}
        
        for (k, features) in self.index.items():
            d = dis.euclidean(queryFeatures, features)
            results[k] = d
        
        results = sorted([(v, k) for (k, v) in results.items()])

        return results        