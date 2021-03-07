open! Balltree
open Torch

(*Examples/ tests *)

(* This one used to cause a crash: leaves can now contain more than one element *)
let bad_points = Tensor.of_float2  [| [| 0.; 0. |]; [| 0.; 0. |] |];;
let bad_points_bt = Balltree.construct_balltree bad_points;;
Stdio.print_string (Balltree.get_string_of_ball bad_points_bt);;

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

(* Test that the balltree can find the nearest point in the 1d case*)
Stdio.print_endline "checking we can retrieve the nearest neighbour in 1d case";
for i=1 to 10 do
    let query_point = Tensor.of_float2 [|[| (Float.of_int i) |]|] in
    let distances, indices = Balltree.query_balltree one_d_bt query_point 1 in
    (* The number found should be i *)
    assert (Float.equal (List.hd distances) 0.);
    (* And also have the correct index (starting from 0) *)
    assert (Int.equal (i-1) (List.hd indices));
done
;;


Stdio.print_endline "checking we retrieve as many neighbours as n_neighbours";
for i=1 to (fst (Tensor.shape2_exn one_d_tensor)) do
    let query_results = (Balltree.query_balltree one_d_bt (Tensor.of_float2 [|[|Float.of_int i|]|]) i) in
    let distances, indices = query_results in
    let l1 = List.length distances in
    let l2 = List.length indices in
    Stdio.print_endline (Int.to_string l1);
    assert ((Int.equal l1 l2) && (Int.equal l2 i))
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
