#require "core_kernel.fheap";;

let compare_leaves (l1: (Tensor.t * Tensor.t)) (l2: (Tensor.t * Tensor.t)) =
    compare_array (Tensor.to_float2_exn (fst l1)) (Tensor.to_float2_exn (fst l2));;

let compare_leaves l1 l2 =
    compare_array (Tensor.to_float2_exn (fst l1)) (Tensor.to_float2_exn (fst l2));;

(* Basically TODO:
    1. Use indices, for the original tensor, everywhere and return those
    2. Write the comparison function
    3. Use it in a heap
*)

let pq = Fheap.create compare;;
let some_heap = List.fold random_list ~init:pq ~f:(fun acc x -> Fheap.add acc x);;



Fheap.to_list some_heap;;


let heap_el = Fheap.add pq (Random.int 10);;
let the_list = random_list |> List.iter ~f:(Fn.compose ignore (Fheap.add pq)) |> Fheap.to_list;;

let the_list =  pq;;
