open Opium
open Core

open Balltree

open! Base
open! Torch
module Model = Bert_model.Distilbert
module Token = Bert_tokenize.Token
module Tokenizer = Bert_tokenize.Bert_tokenizer
module Vocab = Bert_tokenize.Bert_vocab

let vs = Var_store.create ~name:"db" ~device:Cpu () ;;
let model = Model.masked_lm vs Model.Config.base ;;

(* code adapted from ocaml-bert example *)
let get_embedding model tokenizer str = 
    let tokens = Tokenizer.tokenize tokenizer str ~include_special_characters:true in
    let token_ids =
          List.filter_map tokens ~f:(fun token ->
              token.Token.with_id
              |> Option.map ~f:(fun with_id -> with_id.Token.With_id.token_id))
          |> Array.of_list in
    let token_ids = Tensor.of_int2 [| token_ids |] in
    let output = Layer.forward_ model token_ids ~is_training:false in
    let output = (Tensor.get output 0) |> (Tensor.mean1 ~dim:[0] ~dtype:(Torch_core.Kind.T Float) ~keepdim:false) in
    output
;;

let print_string_of_balltree bt _ =
  (Balltree.get_string_of_ball bt)
  (* Printf.sprintf "%s\n" bt *)
  |> Response.of_plain_text
  |> Lwt.return
;;

let get_line_of_file input_path line_idx = 
  In_channel.read_lines ~fix_win_eol:false input_path 
  |> (List.filteri ~f:(fun i _ -> (Int.equal i line_idx)))
  |> List.hd_exn
;;

let get_line_of_file_by_idx_handler fp req = 
  let index = (Router.param req "index") |> Int.of_string in
  get_line_of_file fp index
  |> Response.of_plain_text
  |> Lwt.return
;;


let get_vector_embedding_handler model tokenizer req =
  (Router.param req "string")
  |> (fun x -> (get_embedding model tokenizer x) |> Tensor.to_float1_exn |> (Array.map ~f:Float.to_string) |> (String.concat_array ~sep:"\t"))
  |> Response.of_plain_text
  |> Lwt.return
;;

let get_balltree_nearest_neighbour_handler fp model tokenizer bt req = 
  let n_neighbours = (Router.param req "n") |> Int.of_string in
  let query_string = (Router.param req "string") in
  let query_embedding = query_string |> (get_embedding model tokenizer) in
  let distances, indices = Balltree.query_balltree bt query_embedding n_neighbours in
  let retrieved_lines = List.map indices ~f:(fun index -> (get_line_of_file fp index)) in
  let json : Yojson.Safe.t = `Assoc [ "query", `String query_string; "n_neighbours", `Int n_neighbours; "distances", `List (List.map ~f:(fun x -> `Float x) distances); "indices", `List (List.map ~f:(fun x -> `Int x) indices); "matched_lines", `List (List.map ~f:(fun x -> `String x) retrieved_lines)] in
  Response.of_json json |> Lwt.return
;;
   
let _ =
  let weight_path = Sys.getenv_exn "WEIGHT_PATH" in
  let vocab_path = Sys.getenv_exn "VOCAB_PATH" in
  let embeddings_path = Sys.getenv_exn "EMBEDDINGS_PATH" in
  let plain_textfile_path = Sys.getenv_exn "TEXT_PATH" in
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:weight_path;
  let vocab = Vocab.load ~filename:vocab_path in
  let tokenizer = Tokenizer.create vocab ~lower_case:false in

  let t1 = In_channel.read_lines embeddings_path |> Array.of_list_map ~f:(fun x -> (String.split ~on:'\t' x) |> (Array.of_list_map ~f:Float.of_string)) |> Tensor.of_float2 in
  let bt = Balltree.construct_balltree t1 in
  let _ = Stdio.printf "GB used: %f" (Caml.Gc.allocated_bytes () /. (1024. *. 1024. *. 1024.)) in
  App.empty
  |> App.get "/printbt/" (print_string_of_balltree bt)
  |> App.get "/getline/:index" (get_line_of_file_by_idx_handler plain_textfile_path)
  |> App.get "/getvector/:string" (get_vector_embedding_handler model tokenizer) 
  |> App.get "/getnn/:string/:n" (get_balltree_nearest_neighbour_handler plain_textfile_path model tokenizer bt)
  |> App.run_command
;;
