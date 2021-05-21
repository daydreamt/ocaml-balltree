import argparse
import resource
import sys
import numpy as np
#from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree

parser = argparse.ArgumentParser(usage="")
parser.add_argument('--embeddings', help="e.g. --embeddings good_big_sentence_embeddings.txt", required=True)
args = parser.parse_args()

embeddings = []
with open(args.embeddings) as f:
    for line in f.readlines():
        x = np.array(line.strip().split())
        embeddings.append(x.astype(np.float))

embeddings = np.concatenate(embeddings)
tree = BallTree(embeddings,
                leaf_size=embeddings.shape[0])
print("GB used: {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)))
