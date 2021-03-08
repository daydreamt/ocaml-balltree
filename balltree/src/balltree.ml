(* See https://en.wikipedia.org/wiki/Ball_tree
       http://people.ee.duke.edu/~lcarin/liu06a.pdf , chapter 2
*)
open Core_kernel
open Torch

module Balltree = struct
    (* ENH: Support more distances. Currently only Euclidean. *)
    type ball = { centroid: Tensor.t; dimension: int; radius: float }
    type tree =
        | Leaf of Tensor.t * (int list)
        | Node of (ball * tree * tree)

    (* Returns: a tree structure *)
    let rec construct_balltree_ d d_indices =
        let num_points, num_dim = Tensor.shape2_exn d in
        match num_points with
          | 1 -> Leaf (d, d_indices |> Tensor.to_int1_exn |> Array.to_list)
          | _ ->
            (* The centroid is our median *)
            let median_value, _ = Tensor.median1 d ~dim:0 ~keepdim:false in
            (* Compute the radius: maximum distance of any datapoint 
            ENH: To use other functions, we should probably use the same distance function everywhere
                 and not sometimes Tensor.norm2, and sometimes Tensor.dist *)
            let distances = Tensor.norm2 Tensor.(d - median_value) ~p:(Torch.Scalar.i 2) ~dim:[1] ~keepdim:false in
            let max_distance, _ = Tensor.max2 ~dim:0 ~keepdim:false distances in
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
            let left_indices = Tensor.index_select d_indices ~dim:0 ~index:(less_than_median_indices) in
            let right_subset = Tensor.index_select d ~dim:0 ~index:(greater_than_median_indices) in
            let right_indices = Tensor.index_select d_indices ~dim:0 ~index:(greater_than_median_indices) in
            let num_points_left, _ = Tensor.shape2_exn left_subset in
            let num_points_right, _ = Tensor.shape2_exn right_subset in
            if num_points_right > 0 && num_points_left > 0 then
                Node ({centroid=median_value; dimension=c; radius=max_distance}, construct_balltree_ left_subset left_indices, construct_balltree_ right_subset right_indices)
            else if num_points_right > 0 then
                Leaf (right_subset, right_indices |> Tensor.to_int1_exn |> Array.to_list)
            else
                Leaf (left_subset, left_indices |> Tensor.to_int1_exn |> Array.to_list)


    let construct_balltree d =
        assert (phys_equal (List.length (Tensor.shape d)) 2);
        let d_indices = (Tensor.range ~start:(Torch.Scalar.i 0) ~end_:(Torch.Scalar.i ((fst (Tensor.shape2_exn d) - 1))) ~options:(Torch_core.Kind.T Int64, Torch_core.Device.Cpu)) in
        construct_balltree_ d d_indices


    let unsome = function Some a -> a | None -> raise (Not_found_s (Sexp.of_string "should not happen"))

    (* See https://en.wikipedia.org/wiki/Ball_tree#Pseudocode_2 *)
    let rec query_balltree_ bt pq query_point n_neighbours =
        let top_el_dist = match Fheap.top pq with
            | None -> Float.max_finite_value
            | Some (x, _) -> x in
        match bt with
        | Leaf (d, d_idx) ->
            (*let num_points_leaf, _ = Tensor.shape2_exn d in*)
            let num_points_leaf = List.length d_idx in
            let dist_to_leaf =
                if phys_equal num_points_leaf 1 then
                   (Tensor.dist d query_point) |> Tensor.to_float0_exn
                else
                   (Tensor.dist (Tensor.select ~dim:0 ~index:0 d) query_point) |> Tensor.to_float0_exn in
            if ((Float.compare dist_to_leaf top_el_dist) < 0 ||
                (Fheap.length pq < n_neighbours)) then
                let pq = List.fold d_idx ~init:pq ~f:(fun pq idx -> Fheap.add pq (dist_to_leaf, idx)) in
                if (Fheap.length pq) > n_neighbours then (unsome (Fheap.remove_top pq)) else pq
            else
                pq
        | Node ({centroid=median_value; dimension=c; radius=max_distance_radius}, left, right) ->
            let dist_to_centroid = (Tensor.dist median_value query_point) |> Tensor.to_float0_exn in
            if (((Float.compare (dist_to_centroid -. max_distance_radius) top_el_dist) >= 0) &&
                (Fheap.length pq >= n_neighbours)) then
                pq
            else
                (* OK, promising, find out which of the two children is closer *)
                if (Float.compare (Tensor.get_float1 query_point c) (Tensor.get_float1 median_value c)) < 0 then
                    let pq1 = query_balltree_ left pq query_point n_neighbours in
                    let pq2 = query_balltree_ right pq1 query_point n_neighbours in
                    pq2
                else
                    let pq1 = query_balltree_ right pq query_point n_neighbours in
                    let pq2 = query_balltree_ left pq1 query_point n_neighbours in
                    pq2


    (* sklearn-like API: query_balltree(tree, point, n_neighbours
       returns distances, indices  *)
    let query_balltree bt query_point n_neighbours =
        assert (phys_equal (List.length (Tensor.shape query_point)) 1);
        let create_heap =
            let compare_fun (d1, _) (d2, _) =
                compare_float d2 d1 in
            Fheap.create ~cmp:compare_fun in

        let pq = create_heap in
        let pq_result = query_balltree_ bt pq query_point n_neighbours in
        (* Make sure to only return up to n_neighbours *)
        let ll = List.take (List.rev (Fheap.to_list pq_result)) n_neighbours in
        let distances = List.map ll ~f:fst in
        let indices = List.map ll ~f:snd in
        distances, indices

    let rec get_depth_of_ball d =
        match d with
        | Leaf (_, __) -> 0
        | Node (_, left, right) -> 1 + Int.max (get_depth_of_ball left) (get_depth_of_ball right)

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
        | Leaf (d, d_idx) -> (List.map d_idx ~f:(fun idx -> "=>" ^ spaces ^ (Tensor.to_string d ~line_size:40) ^ "idx: " ^ (Int.to_string idx ^ "\n"))) |> List.to_array |> (String.concat_array ~sep:"\n")
        | Node (_, left, right) ->
            let left_substring = (get_string_of_ball_ left (i + 1) max_depth) in
            let right_substring = (get_string_of_ball_ right (i + 1) max_depth) in
            left_substring ^ "\n" ^ (repeat_string "*" (Int.pow 2 (max_depth - i))) ^ "split number: " ^ (Int.to_string i) ^ (repeat_string "*" (Int.pow 2 (max_depth - i))) ^ "\n" ^ right_substring

    let get_string_of_ball d =
        let depth = get_depth_of_ball d in
        (*     let level_of_indentation = 2 in 
        let n_spaces = depth * level_of_indentation in *)
        get_string_of_ball_ d 0 depth

end
