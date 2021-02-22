import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree

embeddings = []
with open("good_big_sentence_embeddings.txt") as f:
    for line in f.readlines():
        x = np.array(line.strip().split())
        embeddings.append(x.astype(np.float))
        
embeddings = np.concatenate(embeddings)


tree = BallTree(embeddings,
                leaf_size=embeddings.shape[0])
