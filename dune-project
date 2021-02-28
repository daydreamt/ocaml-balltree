(lang dune 2.8)
(name balltree)

(generate_opam_files true)

(authors "Nikolaos")

(maintainers "Nikolaos")
(package
 (name balltree)
 (synopsis "OCaml web framework")
 (description
   "balltree.")
 (depends
  (ocaml
   (>= 4.08))
   torch
   core
   core_kernel
   opium
   ))
