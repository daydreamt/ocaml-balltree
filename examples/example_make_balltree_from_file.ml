open Core
open Torch
open Balltree

let () =

  let cmd_args = Sys.get_argv () in
  if Array.length cmd_args <> 2
  then Printf.failwithf "usage: %s good_big_sentence_embeddings.txt" cmd_args.(0) ();
  let embeddings_path = cmd_args.(1) in
  let t1 = In_channel.read_lines embeddings_path |> Array.of_list_map ~f:(fun x -> (String.split ~on:'\t' x) |> (Array.of_list_map ~f:Float.of_string)) |> Tensor.of_float2 in
  let _ = Balltree.construct_balltree t1 in
  Stdio.printf "GB used: %f\n" (Gc.allocated_bytes () /. (1024. *. 1024. *. 1024.))
