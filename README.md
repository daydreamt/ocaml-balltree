Balltree data structure implemented in ocaml-torch. Currently only supports Euclidean distance. 

Also has an example webserver application, built on top ("balltree as a service").

### Planned:
[] more/ arbitrary distances

[] serialization to disk

[] comparisons with sklearn



### To build and run the example container:
> sudo docker build -t balltree .

> sudo docker run -p 3000:3000 balltree


You could alternatively use data similar to input_sentences.txt by:
> wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt works_of_shakespeare.txt | sort | uniq | shuf | head -n 1000 > input_sentences.txt

We only keep 1000 sentences, since the (distil)bert embeddings are still high dimensional.
> $ du -sh sentence_embeddings.txt
> 574M	sentence_embeddings.txt
