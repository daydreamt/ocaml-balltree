(* See https://en.wikipedia.org/wiki/Ball_tree 
       http://people.ee.duke.edu/~lcarin/liu06a.pdf , chapter 2
*)

#require "torch.toplevel";;
#require "core.top";;
#require "core_kernel.pairing_heap";;
(* There's also core_kernel.fheap *)

open Core;;
open Core_kernel;;
open Torch;;
(* open Tensor;; *)
(* type ball = { centroid: float; radius: float} *)


type ball = { centroid: Tensor.t; dimension: int; radius: float }

(* ENH: More distances *)
(* let dist x y = Tensor.dist x y |> Tensor.to_float0_exn |> (fun x-> x *. (1. /. Float.epsilon_float));; *)

type tree = 
    | Leaf of Tensor.t
    | Node of (ball * tree * tree)
    (*
    | Node of (ball * (Tensor.t) * (Tensor.t))
    *)
;;

(* 
let median arr =
  let len = Array.length arr in
    Array.sort compare arr;
    (array.((len-1)/2) +. array.(len/2)) /. 2.0;;
*)

(* FIXME: return indices everywhere. Just add them to the leaves *)

(* Returns: a tree structure *)
let rec construct_balltree d =
    let num_points, num_dim = Tensor.shape2_exn d in
    match num_points with
      | 1 -> Leaf d (* Leaf (Array.get (Tensor.to_float2_exn d) 0) *)
      | _ -> 
        (* The centroid is our median *)
        let median_value, median_indices = Tensor.median1 d ~dim:0 ~keepdim:false in
        (* Compute the radius: maximum distance of any datapoint ENH: Other functions*)
        let distances = Tensor.norm2 Tensor.(d - median_value) ~p:(Torch.Scalar.i 2) ~dim:[1] ~keepdim:false in
        let max_distance, max_distance_index = Tensor.max2 ~dim:0 ~keepdim:false distances in
        let max_distance = Tensor.to_float0_exn max_distance in
        (* Get dimension of maximum variance *)
        let c = Tensor.std1 d ~dim:[0] ~unbiased:true ~keepdim:false |> Tensor.argmax |> Tensor.to_int0_exn in 
        (* And its value, for the dimension of highest variance *)
        let this_dim_d = Tensor.take d ~index:(Tensor.arange2 ~start:(Scalar.int c) ~step:(Scalar.int num_dim) ~end_:(Scalar.int Int.(num_dim * num_points)) ~options:(Torch_core.Kind.T Int64, Torch_core.Device.Cpu)) in
        (* Get median in this dimension. We will partition the points along it *)
        let this_dim_median = Tensor.median this_dim_d |> Tensor.to_float0_exn in
  
        let less_than_median_indices = Tensor.le this_dim_d (Scalar.f this_dim_median) |> Tensor.nonzero |> Tensor.squeeze in
        let greater_than_median_indices = Tensor.gt this_dim_d (Scalar.f this_dim_median) |> Tensor.nonzero |> Tensor.squeeze in 
        (* Make subsets d into two tensors: one with items less than median in this dimension, one greater or equal in this dimension *)
        let left_subset = Tensor.index_select d ~dim:0 ~index:(less_than_median_indices) in
        let right_subset = Tensor.index_select d ~dim:0 ~index:(greater_than_median_indices) in
        
        Node ({centroid=median_value; dimension=c; radius=max_distance}, construct_balltree left_subset, construct_balltree right_subset)   
;;

let rec query_simple_find_closest_point bt query_point cur_closest cur_dist =
    match bt with 
    | Leaf d -> 
        let dist_to_leaf = (Tensor.dist d query_point) |> Tensor.to_float0_exn in
        if (Float.compare dist_to_leaf cur_dist) < 0 then
            (d, dist_to_leaf)
        else
            (cur_closest, cur_dist)
    | Node ({centroid=median_value; dimension=c; radius=max_distance_radius}, left, right) ->
        let dist_to_centroid = (Tensor.dist median_value query_point) |> Tensor.to_float0_exn in
        if (Float.compare (dist_to_centroid -. max_distance_radius) cur_dist) >= 1 then
            (cur_closest, cur_dist)
        else 
            (* Find out which of the two children is closer *)
            if (Float.compare (Tensor.get_float1 query_point c) (Tensor.get_float1 median_value c)) < 0 then
                query_simple_find_closest_point left query_point median_value dist_to_centroid
            else
                query_simple_find_closest_point right query_point median_value dist_to_centroid   
;;

let rec query_simple_find_closest_point_ bt query_point =
    query_simple_find_closest_point bt query_point query_point (Float.max_finite_value)
;;

(*
let rec query_balltree_ bt point n_neighbours =
    query_simple_find_closest_point bt query_point (Float.max_finite_value)
;;

let query_balltree bt point n_neighbours =
    query_balltree_ bt query_point query_point (Float.max_finite_value)
;;
*) 
  
(* distances, indices = query_balltree(tree, point, n_neighbours=2 *)
let rec get_depth_of_ball d =
    match d with 
    | Leaf d -> 0
    | Node (b, left, right) -> 1 + Int.max (get_depth_of_ball left) (get_depth_of_ball right)

(* Root: first line, center. 
   First print left children, then right.
*)
let rec get_string_of_ball_ d i max_depth =
    let rec repeat_string x i =
        match i with
        | 0 -> x
        | _ -> x ^ (repeat_string x (i - 1)) in
    let spaces = (repeat_string "-" (Int.pow 2 i)) in
    match d with
    | Leaf d -> "=>" ^ spaces ^ (Tensor.to_string d ~line_size:80) ^ "\n"
    | Node (b, left, right) ->
        let left_substring = (get_string_of_ball_ left (i + 1) max_depth) in 
        let right_substring = (get_string_of_ball_ right (i + 1) max_depth) in
        left_substring ^ "\n" ^ (repeat_string "*" (Int.pow 2 (max_depth - i))) ^ "split number: " ^ (Int.to_string i) ^ (repeat_string "*" (Int.pow 2 (max_depth - i))) ^ "\n" ^ right_substring

let get_string_of_ball d = 
    let level_of_indentation = 2 in
    let depth = get_depth_of_ball d in
    let n_spaces = depth * level_of_indentation in
    get_string_of_ball_ d 0 depth
;;


(*Examples *)
let one_d_tensor = Tensor.reshape (Tensor.range ~start:(Torch.Scalar.i 1) ~end_:(Torch.Scalar.i 10) ~options:(Torch_core.Kind.T Float, Torch_core.Device.Cpu))
                                  [-1;1];;
let one_d_tensor_reverse = Tensor.reshape (Tensor.of_float1 ~device:Torch_core.Device.Cpu [|4.;5.;6.;7.;8.;9.;10.;1.;2.;3.;|])
                                  [-1;1];;

Stdio.print_string (get_string_of_ball (construct_balltree one_d_tensor));;
Stdio.print_string (get_string_of_ball (construct_balltree one_d_tensor_reverse));;
assert (String.equal (get_string_of_ball (construct_balltree one_d_tensor)) (get_string_of_ball (construct_balltree one_d_tensor_reverse))) ;;

let one_d_bt = (construct_balltree one_d_tensor);;
query_simple_find_closest_point_ one_d_bt (Tensor.of_float2 [|[|5.3|]|]);;



let tiny_list = [ [1.; 2.;]; [1.; 3.;]; [0.5; 0.3;]; [4.; 5.;] ]
let tiny_array = [| [|1.; 2.;|]; [|1.; 3.;|]; [|0.5; 0.3;|]; [|4.; 5.;|] |]
let tiny_tensor = (Tensor.of_float2 tiny_array);;

let d = tiny_tensor;;
let bt1 = construct_balltree d;;

let bt2 = construct_balltree (Tensor.of_float2 [| [| 1.; 2. |] |]);;
let bt3 = construct_balltree (Tensor.of_float2 [| [| 0.5; 0.3 |] |]);;
construct_balltree (Tensor.of_float2 [| [|1.; 2.;|]; [|1.; 3.;|]; [|4.; 5.;|] |]);;
construct_balltree (Tensor.of_float2  [| [|1.; 2.;|]; [|1.; 3.;|] |]);;
construct_balltree d;;


