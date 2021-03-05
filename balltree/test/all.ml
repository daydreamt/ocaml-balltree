open! Balltree
open Torch

(*Examples/ tests *)

(* This one can cause a crash *)
let bad_points = Tensor.of_float2  [| [| 0.; 0. |]; [| 0.; 0. |] |];;
Balltree.construct_balltree bad_points;;

(* These ones shouldn't *)
let one_d_tensor = Tensor.reshape (Tensor.range ~start:(Torch.Scalar.i 1) ~end_:(Torch.Scalar.i 10) ~options:(Torch_core.Kind.T Float, Torch_core.Device.Cpu)) ~shape:[-1;1];;
let one_d_tensor_reverse = Tensor.reshape (Tensor.of_float1 ~device:Torch_core.Device.Cpu [|4.;5.;6.;7.;8.;9.;10.;1.;2.;3.;|]) ~shape:[-1;1];;

let one_d_bt = Balltree.construct_balltree one_d_tensor;;
let one_d_reverse_bt = Balltree.construct_balltree one_d_tensor_reverse;;
Stdio.print_string (Balltree.get_string_of_ball one_d_bt);;
Stdio.print_string (Balltree.get_string_of_ball one_d_reverse_bt);;
(* It will not be the same, because the indices are different, but the values should be the same *)
(* assert (String.equal (get_string_of_ball one_d_bt) (get_string_of_ball one_d_reverse_bt)) *)

let query_point = (Tensor.of_float2 [|[|5.3|]|]);;

(* Code to help test query_balltree. This does not deal with duplicates. *)
(*
let rec query_simple_find_closest_point bt query_point cur_closest cur_idx cur_dist =
    match bt with
    | Leaf (d, d_idx) ->
        let dist_to_leaf = (Tensor.dist d query_point) |> Tensor.to_float0_exn in
        if (Float.compare dist_to_leaf cur_dist) < 0 then
            (d, d_idx, dist_to_leaf)
        else
            (cur_closest, cur_idx, cur_dist)
    | Node ({centroid=median_value; dimension=c; radius=max_distance_radius}, left, right) ->
        let dist_to_centroid = (Tensor.dist median_value query_point) |> Tensor.to_float0_exn in
        if (Float.compare (dist_to_centroid -. max_distance_radius) cur_dist) >= 1 then
            (cur_closest, cur_idx, cur_dist)
        else
            (* Find out which of the two children is closer *)
            if (Float.compare (Tensor.get_float1 query_point c) (Tensor.get_float1 median_value c)) < 0 then
                query_simple_find_closest_point left query_point median_value cur_idx dist_to_centroid
            else
                query_simple_find_closest_point right query_point median_value cur_idx dist_to_centroid
;;

let rec query_simple_find_closest_point_ bt query_point =
    query_simple_find_closest_point bt query_point query_point (Tensor.of_int1 [|-1|]) (Float.max_finite_value)
;;
let point1, __, dist1 = query_simple_find_closest_point_ one_d_bt query_point;;

let dist2, idx2 = query_balltree one_d_bt (Tensor.of_float2 [|[|5.3|]|]) 1 |> List.hd_exn;;
assert (Float.equal dist1 dist2);;
let point2 = (Tensor.index one_d_tensor [idx2]) |> Tensor.to_float0_exn;;
assert (Float.equal point2 (Tensor.to_float0_exn point1));;
*)

for i=1 to (fst (Tensor.shape2_exn one_d_tensor)) do
    let query_results = (Balltree.query_balltree one_d_bt (Tensor.of_float2 [|[|Float.of_int i|]|]) i) in
    Stdio.print_endline (Int.to_string (List.length query_results));
    assert (Int.equal (List.length query_results) i)
done;;

let tiny_list = [ [1.; 2.;]; [1.; 3.;]; [0.5; 0.3;]; [4.; 5.;] ]
let tiny_array = [| [|1.; 2.;|]; [|1.; 3.;|]; [|0.5; 0.3;|]; [|4.; 5.;|] |]
let tiny_tensor = (Tensor.of_float2 tiny_array);;

let d = tiny_tensor;;
let bt1 = Balltree.construct_balltree d;;

let bt2 = Balltree.construct_balltree (Tensor.of_float2 [| [| 1.; 2. |] |]);;
let bt3 = Balltree.construct_balltree (Tensor.of_float2 [| [| 0.5; 0.3 |] |]);;
Balltree.construct_balltree (Tensor.of_float2 [| [|1.; 2.;|]; [|1.; 3.;|]; [|4.; 5.;|] |]);;
Balltree. construct_balltree (Tensor.of_float2  [| [|1.; 2.;|]; [|1.; 3.;|] |]);;
