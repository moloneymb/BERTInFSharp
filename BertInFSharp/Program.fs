open System
open Modeling

type BertModelTester() = 
    /// Creates a random int32 tensor of the shape within the vocab size.
    static member ids_tensor(shape : int[], vocab_size : int, ?rng : Random, ?name : string) = 
        let rng = defaultArg rng (Random())
        let total_dims = shape |> Array.fold (fun x y -> x*y) 1
        match name with
        | Some(name) -> tf.constant(0, dtype=tf.int32,shape=shape,name=name)
        | _ -> tf.constant(0, dtype=tf.int32,shape=shape)

open Tensorflow

[<EntryPoint>]
let main argv =
    //let xs = [|"";""|]
    //let t = new Tensor(xs |> Array.map (fun x -> System.Text.UTF8Encoding.UTF8.GetBytes(x)),[|xs.LongLength|])

    let batch_size = 2 // TODO check that a batch size of 1 does not cause issues
    let seq_length = 7
    let vocab_size = 99
    let input_ids = BertModelTester.ids_tensor([|batch_size; seq_length|], vocab_size)
    let config = {BertConfig.Default with vocab_size = Some(vocab_size)}
    let bertModel = BertModel(config, true, input_ids)
    tf.get_default_graph().get_operations() |> Seq.filter (fun x -> x.op.OpType = "VariableV2") |> Seq.iter (fun x -> printfn "%s" x.name)
//
//
//    let testText = System.IO.File.ReadAllLines(@"C:\EE\Git\BERTInFSharp\BertInFSharp\sample_text.txt")
//
//    let chkpt = @"C:\Users\moloneymb\Downloads\uncased_L-12_H-768_A-12\uncased_L-12_H-768_A-12"
//    let vocab_file = System.IO.Path.Combine(chkpt, "vocab.txt")
//    let do_lower_case = true
//    let tokenizer =   Tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
//    // TODO figure out stackoverflow
//
//    for line in testText do 
//        printfn "%s" line
//        tokenizer.tokenize (line) |> ignore

    0 

