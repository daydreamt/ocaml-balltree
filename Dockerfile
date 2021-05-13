FROM ocaml/opam:fedora-33-ocaml-4.12 as base

#RUN sudo apk update
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


#WORKDIR ocaml-bert
#ADD . .
#RUN dune build @install

WORKDIR ../learn_search
#RUN pwd;
ADD balltree.opam .
ADD . .
RUN sudo chown -R opam:opam . && \
 opam update && \
 opam install dune core core_kernel base opium ppx_expect ppx_sexp_conv re stdio torch uutf yojson bert && \
 sh -c 'eval `opam config env` dune build @install'


EXPOSE 3000
CMD cd baas; dune exec ./baas.exe -- -p 3000 -d

##RUN opam pin add -yn learn_search . && \
##    opam depext learn_search && \
##    opam install --deps-only learn_search

#ADD . .
#RUN opam install torch core core_kernel opium ppx_sexp_conv bert

#RUN dune build @install

# Install dependencies
#ADD learn_search.opam .
#RUN opam pin add -yn learn_search . && \
#    opam depext learn_search && \
#    opam install --deps-only learn_search

#RUN sh -c 'eval `opam config env` dune build @install'

# Build the app! Note: The chown is somehow necessary, as
# without it the `make build` command will fail with
# permission errors.
#ADD . .
#RUN sudo chown -R opam:nogroup . && \
#    opam config exec make build
#RUN sudo chown -R opam:nogroup . && \
#     sh -c 'eval `opam config env` dune exec ./baas/baas.exe -- -p 3000 -d'
#dune exec ./baas/baas.exe -- -p 3000 -d
