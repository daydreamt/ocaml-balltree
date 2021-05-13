FROM ocaml/opam:fedora-33-ocaml-4.12 as base
#RUN sudo apk add m4 git libffi libffi-dev zlib zlib-dev libev-dev linux-headers zlib-dev libc6-compat
RUN sudo yum install libffi-devel libev-devel libffi zlib zlib-devel -y

RUN opam switch 4.12 && \
 eval $(opam env)

RUN git clone https://github.com/LaurentMazare/ocaml-bert
WORKDIR ocaml-bert
RUN opam pin add -y . && \ 
 opam update && \
 opam install dune && \
  eval $(opam env)
RUN sh -c 'eval `opam config env` dune build .' && \
 sh -c 'eval `opam config env` dune install'

WORKDIR ../learn_search
ADD balltree.opam .
ADD . .
RUN sudo chown -R opam:opam . && \
 opam update && \
 opam install dune core core_kernel base opium ppx_expect ppx_sexp_conv re stdio torch uutf yojson bert && \
sh -c 'eval `opam config env` dune build baas/baas.exe'

FROM base
WORKDIR /app
COPY --from=0 /home/opam/learn_search/_build/default/baas/baas.exe .
COPY --from=0 /home/opam/learn_search/baas/bert-base-uncased-vocab.txt .
COPY --from=0 /home/opam/learn_search/baas/distilbert-base-uncased-rust_model.ot .
COPY --from=0 /home/opam/learn_search/baas/good_first_1000_sentence_embeddings.txt .
COPY --from=0 /home/opam/learn_search/baas/works_of_shakespeare.txt .


CMD WEIGHT_PATH="./distilbert-base-uncased-rust_model.ot" VOCAB_PATH="./bert-base-uncased-vocab.txt" EMBEDDINGS_PATH="./good_first_1000_sentence_embeddings.txt" TEXT_PATH="./works_of_shakespeare.txt" ./baas.exe
EXPOSE 3000

