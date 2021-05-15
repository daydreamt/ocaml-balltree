open! Balltree
open Torch
open Core_kernel
open OUnit2


let test_previously_crashing_ones  _ =
    let odt = Tensor.reshape (Tensor.range ~start:(Torch.Scalar.i 0) ~end_:(Torch.Scalar.i 2) ~options:(Torch_core.Kind.T Float, Torch_core.Device.Cpu)) ~shape:[-1;1] in
    let odt_bt = Balltree.construct_balltree odt in
    let query_point = (Tensor.of_float1 [|Float.of_int 2|]) in
    let (_, _) = Balltree.query_balltree odt_bt query_point 3 in
    (* This one used to cause a crash: leaves can now contain more than one element *)
    let bad_points = Tensor.of_float2  [| [| 0.; 0. |]; [| 0.; 0. |] |] in
    let bad_points_bt = Balltree.construct_balltree bad_points in
    let query_bad_point1 = Tensor.of_float1 [| 0.; 0.; |] in
    let query_bad_point2 = Tensor.of_float1 [| 0.; 0.; |] in
    let distances_bad_1, indices_bad_1 = Balltree.query_balltree bad_points_bt query_bad_point1 2 in
    Stdio.print_endline "Potentially problematic indices:";
    List.iter indices_bad_1 ~f:(fun x -> Stdio.print_endline (Int.to_string x));
    Stdio.print_endline "Their distances:";
    List.iter distances_bad_1 ~f:(fun x -> Stdio.print_endline (Float.to_string x));
    let distances_bad_2, indices_bad_2 = Balltree.query_balltree bad_points_bt query_bad_point2 2 in
    Stdio.print_endline "Potentially problematic indices:";
    List.iter indices_bad_2 ~f:(fun x -> Stdio.print_endline (Int.to_string x));
    Stdio.print_endline "Their distances:";
    List.iter distances_bad_2 ~f:(fun x -> Stdio.print_endline (Float.to_string x));

    assert_equal true ((List.is_sorted ~compare:Float.compare distances_bad_1) &&  (List.is_sorted ~compare:Float.compare distances_bad_2))


let test_one_d_single_reverse _ =
    let one_d_tensor = Tensor.reshape (Tensor.range ~start:(Torch.Scalar.i 1) ~end_:(Torch.Scalar.i 10) ~options:(Torch_core.Kind.T Float, Torch_core.Device.Cpu)) ~shape:[-1;1] in
    let one_d_tensor_reverse = Tensor.reshape (Tensor.of_float1 ~device:Torch_core.Device.Cpu [|4.;5.;6.;7.;8.;9.;10.;1.;2.;3.;|]) ~shape:[-1;1] in
    let one_d_bt = Balltree.construct_balltree one_d_tensor in
    let one_d_reverse_bt = Balltree.construct_balltree one_d_tensor_reverse in
    Stdio.print_string (Balltree.get_string_of_ball one_d_bt);
    Stdio.print_string (Balltree.get_string_of_ball one_d_reverse_bt);
    (* It will not be the same, because the indices are different, but the values should be the same *)
    (* assert (String.equal (get_string_of_ball one_d_bt) (get_string_of_ball one_d_reverse_bt)) *)

    let distances, indices = (Balltree.query_balltree one_d_bt (Tensor.of_float1 [|6.|]) 4) in
    let _ = indices in
    assert_equal true (List.is_sorted ~compare:Float.compare distances)


let check_if_single_i_one_d_returns_n i =
    let one_d_tensor = Tensor.reshape (Tensor.range ~start:(Torch.Scalar.i 1) ~end_:(Torch.Scalar.i 10) ~options:(Torch_core.Kind.T Float, Torch_core.Device.Cpu)) ~shape:[-1;1] in
    let query_point = Tensor.of_float1 [| (Float.of_int i) |] in
    let one_d_bt = Balltree.construct_balltree one_d_tensor in
    let distances, indices = Balltree.query_balltree one_d_bt query_point 1 in
    (* The number found should be i *)
    (* Stdio.print_endline ("i: " ^ (Int.to_string i));
    Stdio.print_endline ("Number of distances returned: " ^ (Int.to_string (List.length distances)));
    List.iter distances ~f:(fun x -> Stdio.print_string ((Float.to_string x) ^ " "));
    List.iter indices ~f:(fun x -> Stdio.print_string ((Int.to_string x) ^ " "));
    Stdio.print_endline "";
    Stdio.print_endline ("closest distance: " ^ (Float.to_string (List.hd_exn distances)));
    *)
    ((Float.equal (List.hd_exn distances) 0.) && (Int.equal (i-1) (List.hd_exn indices)) && (List.is_sorted ~compare:Float.compare distances))
    
let test_number_neighbours1 _ =
    let all_good = 
        List.map ~f:check_if_single_i_one_d_returns_n (List.range 1 10 ~stop:`inclusive)
        |> List.reduce_exn ~f:(fun x y -> x && y) in
        assert_equal true all_good

let check_if_single_i_one_d_returns_n_2 one_d_bt i =
    let query_results = (Balltree.query_balltree one_d_bt (Tensor.of_float1 [|Float.of_int i|]) i) in
    let distances, indices = query_results in
    let l1 = List.length distances in
    let l2 = List.length indices in
    (*
    Stdio.print_endline ("distances for " ^ (Int.to_string i));
    List.iter distances ~f:(fun x -> Stdio.print_string ((Float.to_string x) ^ " "));
    Stdio.print_endline ("\nindices for " ^ (Int.to_string i));
    List.iter indices ~f:(fun x -> Stdio.print_string ((Int.to_string x) ^ " "));
    Stdio.print_endline "\nprint tree";
    Stdio.print_endline (Balltree.get_string_of_ball one_d_bt);
    *)
    (List.is_sorted ~compare:Float.compare distances) && ((Int.equal l1 l2) && (Int.equal l2 i))
    
let test_number_neighbours2 _ =
    Stdio.print_endline "checking we retrieve as many neighbours as n_neighbours";
    let one_d_tensor = Tensor.reshape (Tensor.range ~start:(Torch.Scalar.i 1) ~end_:(Torch.Scalar.i 10) ~options:(Torch_core.Kind.T Float, Torch_core.Device.Cpu)) ~shape:[-1;1] in
    let one_d_bt = Balltree.construct_balltree one_d_tensor in
    let n = fst (Tensor.shape2_exn one_d_tensor) in
    let all_good = 
        List.map ~f:(fun i -> check_if_single_i_one_d_returns_n_2 one_d_bt i) (List.range 1 n ~stop:`inclusive)
        |> List.reduce_exn ~f:(fun x y -> x && y) in
        assert_equal true all_good        

let test_is_sorted1 _ = 
    let bt0 = Balltree.construct_balltree (Tensor.of_float2 [| [|0.; 0.;|]; |]) in
    let distances0, _ = Balltree.query_balltree bt0 (Tensor.of_float1 [| 0.; 1.;|]) 1 in
    assert_equal true (List.is_sorted ~compare:Float.compare distances0)

let test_is_sorted2 _ =
    let bt2 = Balltree.construct_balltree (Tensor.of_float2 [| [|0.; 0.;|]; [| 1.; 2. |] |]) in
    let distances2, _ = Balltree.query_balltree bt2 (Tensor.of_float1 [| 0.; 1.;|]) 1 in
    Stdio.print_endline (Float.to_string (List.hd_exn distances2));
    assert_equal true (List.is_sorted ~compare:Float.compare distances2)

let test_euclidean_distances _ = 
    let point3_array = [| [|1.; 2.;|]; [|1.; 3.;|]; [|4.; 5.;|] |] in
    let query_point3 = (Tensor.of_float1 [| 0.; 0.;|]) in
    let query_point3_dimlist = Tensor.shape query_point3 in
    Stdio.print_endline ("Dimensions of query_point3:" ^ (query_point3_dimlist |> List.map ~f:Int.to_string |> (String.concat~sep:"\t"))) ;

    let real_distances3 = Array.map point3_array ~f:(fun x -> Tensor.dist (Tensor.of_float1 x) query_point3) |> Array.map ~f:(Tensor.to_float0_exn) in
    let bt3 = Balltree.construct_balltree (Tensor.of_float2 point3_array) in
    let distances3, _ = Balltree.query_balltree bt3 query_point3 3 in
    let zeroes = Array.map2_exn (distances3 |> Array.of_list) real_distances3 ~f:(Float.sub) in
    let array_sum = Array.reduce_exn zeroes ~f:(fun x y -> x +. y) in
    assert_equal true ((Float.compare (Float.abs array_sum) Float.epsilon_float) < 0)


let suite =
    "TestBalltrees" >::: [
        "test_previously_crashing_cases" >:: test_previously_crashing_ones;
        "test_one_d_single_reverse" >:: test_one_d_single_reverse;
        "test_n_returned1" >:: test_number_neighbours1;
        "test_n_returned2" >:: test_number_neighbours2;
        "test_is_sorted1" >:: test_is_sorted1;
        "test_is_sorted2" >:: test_is_sorted2;
        "test_euclidean_distances" >:: test_euclidean_distances;
    ]
let () =
    run_test_tt_main suite
