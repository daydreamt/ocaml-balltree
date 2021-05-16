Balltree data structure implemented in ocaml-torch. Currently only supports Euclidean distance. 

Also has an example webserver application, built on top ("balltree as a service").

### Planned:
[] more/ arbitrary distances

[] serialization to disk

[] comparisons with sklearn



### To build and run the example container:
> sudo docker build -t balltree .

> sudo docker run -p 3000:3000 balltree

> $ curl http://localhost:3000/getnn/Right%20noble%20Burgundy,/10
```json 
{"query":"Right noble Burgundy,","n_neighbours":10,"distances":[0.0,118.38802337646484,122.64796447753906,124.86466217041016,125.55013275146484,131.27127075195312,134.02183532714844,135.88157653808594,136.78036499023438,136.79348754882812],"indices":[106,267,66,701,428,20,270,599,779,728],"matched_lines":["Right noble Burgundy,\r","By Providence divine.\r","another.\r","CLOWN, Servant to Othello\r","IAGO, his Ancient\r","                       his BRETHREN\r","Tis true.\r","      Who, Hero?\r","Alas, Iago,\r","By whom, Aeneas?\r"]}
```

You could alternatively use data similar to input_sentences.txt by:
> wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt works_of_shakespeare.txt | sort | uniq | shuf | head -n 1000 > input_sentences.txt

We only keep 1000 sentences, since the (distil)bert embeddings are still high dimensional.
> $ du -sh sentence_embeddings.txt

> 574M	sentence_embeddings.txt
