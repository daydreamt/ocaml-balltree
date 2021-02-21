(* This file is derivative work of https://github.com/LaurentMazare/ocaml-bert/blob/master/src/model/distilbert.ml 
    so it has the same license
*)
#require "core";;
#require "torch";;
#require "bert";;
#require "bert.tokenize";;
#require "bert.model";;

open Core
open! Base
open! Torch
module Model = Bert_model.Distilbert
module Token = Bert_tokenize.Token
module Tokenizer = Bert_tokenize.Bert_tokenizer
module Vocab = Bert_tokenize.Bert_vocab


let vs = Var_store.create ~name:"db" ~device:Cpu () ;;
let model = Model.masked_lm vs Model.Config.base ;;
let weight_path = "distilbert-base-uncased-rust_model.ot" ;;

Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:weight_path;;

let vocab_path = "bert-base-uncased-vocab.txt" ;;
let vocab = Vocab.load ~filename:vocab_path ;;



let tokenizer = Tokenizer.create vocab ~lower_case:true ;;

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

(* FIXME: command line argument *)
(* 
let input_path = "sentences.txt" ;;
let output_path = "sentence_embeddings.txt" ;;
*)
let input_path = "/home/dd/Downloads/works_of_shakespeare.txt" ;;
let output_path = "big_sentence_embeddings.txt" ;;

(* 
In_channel.read_lines input_path |> List.map ~f:(fun x -> (get_embedding model tokenizer x) |> Tensor.to_float1_exn |> (Array.map ~f:Float.to_string) |> (String.concat_array ~sep:"\t")) |> Out_channel.write_lines output_path;;
*)

Out_channel.with_file
  ~append:true
  output_path
  ~f:(fun outc -> In_channel.read_lines input_path |> List.iter ~f:(fun x -> (get_embedding model tokenizer x) |> Tensor.to_float1_exn |> (Array.map ~f:Float.to_string) |> (String.concat_array ~sep:"\t") |> fprintf outc "%s\n"))

