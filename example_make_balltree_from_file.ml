#use "balltree.ml"

let embeddings_path = "good_big_sentence_embeddings.txt" ;;

(* In_channel.read_lines embeddings_path |> List.map ~f:(fun x -> (String.split ~on:'\t' x) |> (Array.of_list_map ~f:Float.of_string) |> Tensor.of_float1) ;; *)

let t1 = In_channel.read_lines embeddings_path |> Array.of_list_map ~f:(fun x -> (String.split ~on:'\t' x) |> (Array.of_list_map ~f:Float.of_string)) |> Tensor.of_float2;;
let bt1 = construct_balltree t1;;





