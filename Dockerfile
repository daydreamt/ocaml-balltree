FROM ocaml/opam:alpine

WORKDIR learn_search

ADD balltree.opam

# Install dependencies
#ADD learn_search.opam .
#RUN opam pin add -yn learn_search . && \
#    opam depext learn_search && \
#    opam install --deps-only learn_search

RUN dune build @install

# Build the app! Note: The chown is somehow necessary, as
# without it the `make build` command will fail with
# permission errors.
ADD . .
#RUN sudo chown -R opam:nogroup . && \
#    opam config exec make build
RUN sudo chown -R opam:nogroup . && \
     dune exec ./baas/baas.exe -- -p 3000 -d
