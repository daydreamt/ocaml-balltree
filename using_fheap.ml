#require "core_kernel.fheap";;

let compare_leaves (l1: (Tensor.t * Tensor.t)) (l2: (Tensor.t * Tensor.t)) =
    compare_array (Tensor.to_float2_exn (fst l1)) (Tensor.to_float2_exn (fst l2));;

let compare_leaves l1 l2 =
    compare_array (Tensor.to_float2_exn (fst l1)) (Tensor.to_float2_exn (fst l2));;

(* Basically TODO:
    1. Use indices, for the original tensor, everywhere and return those
    2. Write the comparison function
    
        the comparison function depends on the query item!!!
    3. Use it in a heap
*)

(* ENH: allow dist function to be passed also *)
(* method 1: should the heap recalculate the distances? *)
let create_heap_ query_item = 
    let compare_fun (x1, _) (x2, _) =
        compare_float ((Tensor.dist query_item x1) |> Tensor.to_float0_exn)
                      ((Tensor.dist query_item x1) |> Tensor.to_float0_exn) in
    Fheap.create compare_fun;;

(* Method 2: should the heap just compare existing distances?
   Note the compare_float d2 d1, as we want to pop the maximum elements
   and keep the minimum elements. *)
let create_heap query_item = 
    let compare_fun (d1, _) (d2, _) =
        compare_float d2 d1 in
    Fheap.create compare_fun;;


let pq = create_heap query_point;;
let pq2 = Fheap.add pq (closest_node_distance, closest_node_idx);;
let pq3 = Fheap.add pq2 (closest_node_distance +. 1., closest_node_idx);;

let l =  
   match (Fheap.remove_top pq3) with 
   | Some x -> (Fheap.to_list x)
   | None -> []

;;


let some_heap = List.fold random_list ~init:pq ~f:(fun acc x -> Fheap.add acc x);;



Fheap.to_list some_heap;;


let heap_el = Fheap.add pq (Random.int 10);;
let the_list = random_list |> List.iter ~f:(Fn.compose ignore (Fheap.add pq)) |> Fheap.to_list;;

let the_list =  pq;;
