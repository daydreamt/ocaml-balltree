import argparse
import resource
import os
import sys
import numpy as np
#from scipy.sparse import csr_matrix
import psutil
from sklearn.neighbors import BallTree

parser = argparse.ArgumentParser(usage="")
parser.add_argument('--embeddings', help="e.g. --embeddings good_big_sentence_embeddings.txt", required=True)
args = parser.parse_args()

embeddings = []
with open(args.embeddings) as f:
    for line in f.readlines():
        x = np.array(line.strip().split())
        embeddings.append(x.astype(np.float))

embeddings = np.stack(embeddings)
tree = BallTree(embeddings,
                leaf_size=1)
process = psutil.Process(os.getpid())
print("GB used: {}".format(process.memory_info().rss / (1024**3)))
