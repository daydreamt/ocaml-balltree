(lang dune 2.8)
(name balltree)
(version 0.0.1)

(generate_opam_files true)

(source (github daydreamt/balltree.ml))

(authors "Nikolaos")
(maintainers "Nikolaos")

(package
 (name balltree)
 (synopsis "OCaml balltree implementation")
 (description
   "balltree.")
 (depends
  (ocaml
   (>= 4.08))
   torch
   core
   core_kernel.fheap
   opium
   ppx_sexp_conv
   ))
