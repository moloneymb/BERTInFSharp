
#I @"C:\Users\moloneymb\.nuget\packages\"
#r @"system.runtime.compilerservices.unsafe\4.5.2\lib\netstandard2.0\System.Runtime.CompilerServices.Unsafe.dll"
#r @"numsharp\0.20.5\lib\netstandard2.0\NumSharp.Core.dll"
#r @"tensorflow.net\0.14.0\lib\netstandard2.0\TensorFlow.NET.dll"
#r @"system.memory\4.5.3\lib\netstandard2.0\System.Memory.dll"
#r @"google.protobuf\3.10.1\lib\netstandard2.0\Google.Protobuf.dll"
#r @"argu\6.0.0\lib\netstandard2.0\Argu.dll"
#r @"csvhelper\12.2.3\lib\net47\CsvHelper.dll"
#r @"newtonsoft.json\12.0.2\lib\net45\Newtonsoft.Json.dll"

#load @"..\BertInFSharp\utils.fs"
#load @"..\BertInFSharp\tokenization.fs"
#load @"..\BertInFSharp\run_classifier.fs"
#load @"..\BertInFSharp\modeling.fs"

open Tokenization
open System
open System.IO
open Newtonsoft.Json.Linq
open Modeling
open Tensorflow.Operations.Activation
open Modeling.Activation
open NumSharp
open Tensorflow
open System.Collections.Generic

let tf = Tensorflow.Binding.tf

let sess = tf.Session()

let chkpt = @"C:\Users\moloneymb\Downloads\uncased_L-12_H-768_A-12\uncased_L-12_H-768_A-12"

let vocab_file = Path.Combine(chkpt, "vocab.txt")
let bert_config_file = Path.Combine(chkpt, "bert_config.json")

let do_lower_case = true
let tokenizer =   Tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

let bert_config = BertConfig.from_json_string(File.ReadAllText(bert_config_file))

//bert_config 

let batch_size = 1
let MAX_SEQ_LENGTH = 128

let vocab = File.ReadAllLines(vocab_file)

let movie_reviews = 
    [|
        "a stirring , funny and finally transporting re imagining of beauty and the beast and 1930s horror films",1
        "apparently reassembled from the cutting room floor of any given daytime soap",0
        "they presume their audience won't sit still for a sociology lesson",0
        "this is a visually stunning rumination on love , memory , history and the war between art and commerce",1
        "jonathan parker 's bartleby should have been the be all end all of the modern office anomie films",1
    |]

open RunClassifier

let mm = movie_reviews |> Array.map (fun (x,y) -> InputExample(text_a = x, label = string y) :> IExample)

let examples = RunClassifier.convert_examples_to_features(mm,vocab,MAX_SEQ_LENGTH, tokenizer :> Tokenization.ITokenizer)

open Tensorflow

let input_ids = tf.placeholder(tf.int32,TensorShape([|batch_size; MAX_SEQ_LENGTH|]))
let input_mask = tf.placeholder(tf.int32,TensorShape([|batch_size; MAX_SEQ_LENGTH|]))

let bertModel = BertModel(bert_config, false, input_ids=input_ids,input_mask=input_mask)

type tensorflow with
    member this.constant(values : string[], ?shape : int[], ?name : string) =
        let name = defaultArg name "Const"
        let g = ops.get_default_graph()
        let tensor_proto = 
            let tp = TensorProto()
            tp.Dtype <- tf.string.as_datatype_enum()
            tp.StringVal.AddRange(values |> Array.map (fun x -> Google.Protobuf.ByteString.CopyFromUtf8(x)))
            tp.TensorShape <- tensor_util.as_shape(defaultArg shape [|values.Length|])
            tp
        let attrs = Dictionary([|"value",AttrValue(Tensor = tensor_proto); "dtype", AttrValue(Type = tensor_proto.Dtype)|] |> dict)
        g.create_op("Const",[||],[|tf.string|],attrs = attrs, name = name).output

    member this.all_variable_names() = 
        let graph = tf.get_default_graph()
        [| for x in graph.get_operations() do if x.op.OpType = "VariableV2" then yield x.op.name|]

    member this.restore(path : string, ?variable_names : string[], ?mapping : string -> string, ?name : string) = 
        let name = defaultArg name "restore"
        let mapping = defaultArg mapping id
        let graph = tf.get_default_graph()
        let variable_names = variable_names |> Option.defaultWith (fun _ -> this.all_variable_names())
        let variables = [| for x in variable_names  -> graph.get_operation_by_name(mapping(x))|]
        let dataTypes = [| for x in variables -> x.op.output.dtype.as_base_dtype()|]
        // TODO proper slices requires making an extra C shim to expose types and mapping
        let restore = gen_ops.restore_v2(tf.constant(path), 
                                         tf.constant(variable_names),
                                         tf.constant(Array.create variables.Length ""),
                                         dataTypes)
        let assignOps = [|for r,v in (restore,variables) ||> Array.zip -> tf.assign(v.output,r)|]
        tf.group(assignOps,name=name)

    // Not tested yet
    member this.save(path : string, ?variableNames : string[], ?name : string) = 
        let name = defaultArg name "save"
        let graph = tf.get_default_graph()
        let variable_names = variableNames |> Option.defaultWith (fun _ -> this.all_variable_names())
        let variables= variable_names |> Array.map (fun x -> graph.get_operation_by_name(x).output)
        gen_ops.save_v2(tf.constant(path),
                        tf.constant(variable_names),
                        tf.constant(Array.create variables.Length ""),
                        variables, 
                        name = name)


//# Use "pooled_output" for classification tasks on an entire sentence.
//# Use "sequence_outputs" for token-level output.
//output_layer = bert_outputs["pooled_output"]

let restore = tf.restore(Path.Combine(chkpt,"bert_model.ckpt"))
sess.run(restore)

let t1 = NDArray([|examples.[0].input_ids|])
let t2 = NDArray([|examples.[0].input_mask|])

/// [-0.791169226f; -0.372503042f; -0.784386933f; 0.597510815f;  
let res = sess.run(bertModel.PooledOutput,[|FeedItem(input_ids,t1); FeedItem(input_mask,t2)|])


////let matmuls = 
////    [| for op in  graph.get_operations() do
////        if op.op.OpType = "MatMul" then 
////            yield op.op |]
////
////let muls = 
////    [| for op in  graph.get_operations() do
////        if op.op.OpType = "Mul" then 
////            yield op.op |]
//
//let nameFilter (name : string) = 
//    name.StartsWith("bert/") && 
//    (not(name.Contains("nitializer"))) && 
//    (not(name.Contains("Identity"))) && 
//    (not(name.Contains("Assert"))) && 
//    (not(name.Contains("Assign"))) && 
//    (not(name.Contains("assert"))) && 
//    (not(name.Contains("Const")))
//
//let cleanupName (name : string) = 
//    name.Replace("Variance","variance")
//        .Replace("MatMul/MatMul","MatMul")
//        .Replace("MatMul_1/MatMul","MatMul_1")
//
//let filtered_ops = 
//    [| 
//        for x in graph.get_operations() do 
//            if nameFilter(x.name)
//            then 
//                if x.outputs.Length > 0 
//                then 
//                    yield x
//    |]
//
//filtered_ops.Length
//
//let outputs = sess.run(filtered_ops |> Array.map (fun x -> x.outputs.[0]), [|FeedItem(input_ids,t1); FeedItem(input_mask,t2)|])
//
//let shapeToString(xs : int32[]) = 
//    match xs with 
//    | [||] -> "()"
//    | [|x|] -> sprintf "(%i,)" x
//    | xs -> xs |> Array.map string |> String.concat ", " |> sprintf "(%s)"
//
//let lines = 
//    [| 
//        for y,x in (outputs,filtered_ops) ||> Array.zip do
//            if nameFilter(x.name) then
//                let value = 
//                    if y.dtype.Name = np.int32.Name
//                    then y.Data<int32>().ToArray() |> Array.map string |> String.concat ", " |> sprintf "[%s]"
//                    elif y.dtype.Name = np.float32.Name
//                    then sprintf "%.1f" (y.Data<float32>().ToArray() |> Array.sum)
//                    else ""
//                yield sprintf "%s %s %s" (cleanupName(x.name)) (y.shape |> shapeToString) value
//    |]
//
//
//File.WriteAllLines(@"C:\EE\fsharpOps.txt", lines )
//
//outputs.[0].sum().ToString()
//
////for x in [| for x in graph.get_operations() do if x.name.StartsWith("bert/") then yield x.name |] do 
////    printfn "%s" x
//
////tf.train.init_from_checkpoint()
//
////muls.Length
//// TODO - find out where it goes wrong
////sess.run(graph.get_operation_by_name(vs.[1]).output).Data<float32>() |> Seq.toArray |> Array.exists (fun x -> System.Single.IsNaN(x))
//
////sess.run(matmuls.[1].output, [|FeedItem(input_ids,t1); FeedItem(input_mask,t2)|]).shape
//
////matmuls.[1].name
////sess.run(muls.[3].output, [|FeedItem(input_ids,t1); FeedItem(input_mask,t2)|]).shape
////muls.[3].name
////matmuls.[1].inputs.[0].op.inputs.[0].op.inputs.[0].op.OpType
////sess.run(matmuls.[56].output, [|FeedItem(input_ids,t1); FeedItem(input_mask,t2)|])
////sess.run(matmuls.[58].output, [|FeedItem(input_ids,t1); FeedItem(input_mask,t2)|])
//
//sess.run("bert/encoder/layer_5/attention/self/value/bias:0").[0].Data<float32>() |> Seq.toArray
//
//// xx
/////matmuls.[2]
//
//[|for i in  t1.[0].Data<int32>() -> i |]
////[|101; 1037; 18385; 1010; 6057; 1998; 2633; 18276; 2128; 16603; 1997; 5053;
////    1998; 1996; 6841; 1998; 5687; 5469; 3152; 102; 0; ...
//[|for i in t2.[0].Data<int32>() -> i |]
////[|1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0;
//
//res
//t1
//t2
//
//t1.shape
//[|for x in res.[0].Data<float32>().ToArray() -> sprintf "%0.3f" x|] |> String.concat " "
//
//let xx =  "0.018 0.030 -0.123 0.204 0.146 -0.007 0.094 -0.108 0.024 -0.030 -0.054 -0.305 0.130 0.114 -0.095 -0.005 0.053 0.037 -0.107 -0.178 -0.013"
//
//
////let input_ids = BertModelTester.ids_tensor([|batch_size; seq_length|], vocab_size)
////    let batch_size = 2 // TODO check that a batch size of 1 does not cause issues
////    let seq_length = 7
////    let vocab_size = 99
////    let input_ids = BertModelTester.ids_tensor([|batch_size; seq_length|], vocab_size)
////    let config = {BertConfig.Default with vocab_size = Some(vocab_size)}
////    let bertModel = BertModel(config, true, input_ids)
////    tf.get_default_graph().get_operations() |> Seq.filter (fun x -> x.op.OpType = "VariableV2") |> Seq.iter (fun x -> printfn "%s" x.name)
//
////let bert_model = BertModel(bert_config, false,)
//
////"https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1.tar.gz"
//
//// TODO get model automatically 
//// TODO ungzip / untar
//
///// Get the vocab file and casing info from the Hub module."""
////let create_tokenizer_from_hub_module() = 
//  //with tf.Graph().as_default():
//  //  bert_module = hub.Module(BERT_MODEL_HUB)
//  //  tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
//  //  with tf.Session() as sess:
//  //    vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
//  //                                          tokenization_info["do_lower_case"]])
//  //    
//
////tokenizer = create_tokenizer_from_hub_module()

