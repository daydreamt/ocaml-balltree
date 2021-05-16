(* get_embedding function is essentially derivative work of https://github.com/LaurentMazare/ocaml-bert/blob/master/src/model/distilbert.ml
*)

open Core
open! Base
open! Torch
module Model = Bert_model.Distilbert
module Token = Bert_tokenize.Token
module Tokenizer = Bert_tokenize.Bert_tokenizer
module Vocab = Bert_tokenize.Bert_vocab

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


let () = 
    let vs = Var_store.create ~name:"db" ~device:Cpu () in
    let model = Model.masked_lm vs Model.Config.base in
    let cmd_args = Sys.get_argv () in
    if Array.length cmd_args <> 5
           then Printf.failwithf "usage: %s distilbert-base-uncased-rust_model.ot bert-base-uncased-vocab.txt input_sentences.txt big_sentence_embeddings.txt" cmd_args.(0) ();
    let weight_path = cmd_args.(1) in
    let vocab_path = cmd_args.(2) in
    let input_path = cmd_args.(3) in
    let output_path = cmd_args.(4) in
    Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:weight_path;
    let vocab = Vocab.load ~filename:vocab_path in
    let tokenizer = Tokenizer.create vocab ~lower_case:false in

    Out_channel.with_file
      ~append:false
      output_path
      ~f:(fun outc -> In_channel.read_lines input_path |> List.iter ~f:(fun x -> (get_embedding model tokenizer x) |>   Tensor.to_float1_exn |> (Array.map ~f:Float.to_string) |> (String.concat_array ~sep:"\t") |> fprintf outc "%s\n"))
