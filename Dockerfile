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

WORKDIR ../ocaml-balltree
ADD balltree.opam .
ADD . .
RUN sudo chown -R opam:opam . && \
 opam update && \
 opam install dune core core_kernel base opium ppx_expect ppx_sexp_conv re stdio torch uutf yojson bert && \
sh -c 'eval `opam config env` dune build baas/baas.exe'

FROM base
WORKDIR /app
COPY --from=0 /home/opam/ocaml-balltree/_build/default/baas/baas.exe .
COPY --from=0 /home/opam/ocaml-balltree/baas/bert-base-uncased-vocab.txt .
COPY --from=0 /home/opam/ocaml-balltree/baas/distilbert-base-uncased-rust_model.ot .
COPY --from=0 /home/opam/ocaml-balltree/baas/sentence_embeddings.txt .
COPY --from=0 /home/opam/ocaml-balltree/baas/input_sentences.txt .


CMD WEIGHT_PATH="./distilbert-base-uncased-rust_model.ot" VOCAB_PATH="./bert-base-uncased-vocab.txt" EMBEDDINGS_PATH="./sentence_embeddings.txt" TEXT_PATH="./input_sentences.txt" ./baas.exe
EXPOSE 3000

