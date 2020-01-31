////"https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1.tar.gz"
//// TODO get model automatically 
//// TODO ungzip / untar

#I @"C:\Users\moloneymb\.nuget\packages\"
#r @"system.runtime.compilerservices.unsafe\4.5.2\lib\netstandard2.0\System.Runtime.CompilerServices.Unsafe.dll"
#r @"numsharp\0.20.5\lib\netstandard2.0\NumSharp.Core.dll"
#r @"tensorflow.net\0.14.0\lib\netstandard2.0\TensorFlow.NET.dll"
#r @"system.memory\4.5.3\lib\netstandard2.0\System.Memory.dll"
#r @"google.protobuf\3.10.1\lib\netstandard2.0\Google.Protobuf.dll"
#r @"argu\6.0.0\lib\netstandard2.0\Argu.dll"
#r @"csvhelper\12.2.3\lib\net47\CsvHelper.dll"
#r @"newtonsoft.json\12.0.2\lib\net45\Newtonsoft.Json.dll"
#r @"sharpziplib\1.2.0\lib\net45\ICSharpCode.SharpZipLib.dll"

#load @"..\BertInFSharp\common.fs"
#load @"..\BertInFSharp\utils.fs"
#load @"..\BertInFSharp\tokenization.fs"
#load @"..\BertInFSharp\run_classifier.fs"
#load @"..\BertInFSharp\modeling.fs"
#load @"..\BertInFSharp\optimization.fs"

#time "on"

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
open RunClassifier
open Common

let tf = Tensorflow.Binding.tf

let do_lower_case = true
let tokenizer =   Tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
let bert_config = BertConfig.from_json_string(File.ReadAllText(bert_config_file))
// Compute train and warmup steps from batch size
// These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)

let BATCH_SIZE = 2 // when not on laptop bump up to 32
let NUM_LABELS = 2
let LEARNING_RATE = 2e-5f
let MAX_SEQ_LENGTH = 128
let NUM_TRAIN_EPOCHS = 3.0f
// Warmup is a period of time where hte learning rate 
// is small and gradually increases--usually helps training.
let WARMUP_PROPORTION = 3.0f
// Model configs
let SAVE_CHECKPOINTS_STEPS = 500
let SAVE_SUMMARY_STEPS = 100

let vocab = File.ReadAllLines(vocab_file)

open ICSharpCode.SharpZipLib.GZip
open ICSharpCode.SharpZipLib.Tar

let dataset = @"C:\EE\aclImdb\"

let download_and_extract() =
    let wc = new Net.WebClient()
    wc.DownloadFile(@"http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", @"C:\EE\aclImdb_v1.tar.gz")
    let extractTGZ(gzArchiveName : string, destFolder : string) =
        use inStream = File.OpenRead(gzArchiveName)
        use gzipStream = new GZipInputStream(inStream)
        use tarArchive = TarArchive.CreateInputTarArchive(gzipStream)
        tarArchive.ExtractContents(destFolder)
    extractTGZ(@"C:\EE\aclImdb_v1.tar.gz", dataset)

let getTrainTest limit = 
    let vocab_map = vocab |> Array.mapi (fun i x -> (x,i)) |> Map.ofArray
    let f x y v = 
        Directory.GetFiles(Path.Combine(dataset, "aclImdb",x,y)) 
        |> Array.truncate limit
        |> Async.mapiChunkBySize 200 (fun _ x -> InputExample(text_a = File.ReadAllText(x), label = string v) :> IExample)
    let g x = 
        let mm = [| yield! f x "pos" 1; yield! f x "neg" 0|] |> Array.shuffle
        convert_examples_to_features(mm,vocab_map,MAX_SEQ_LENGTH, tokenizer :> Tokenization.ITokenizer)
    (g "train", g "test")


let train,test = getTrainTest 2500


let input_ids = tf.placeholder(tf.int32,TensorShape([|BATCH_SIZE; MAX_SEQ_LENGTH|]))
let input_mask = tf.placeholder(tf.int32,TensorShape([|BATCH_SIZE; MAX_SEQ_LENGTH|]))
let labels = tf.placeholder(tf.int32,TensorShape([|BATCH_SIZE|]))
let bertModel = BertModel(bert_config, false, input_ids = input_ids, input_mask = input_mask)


// Use "pooled_output" for classification tasks on an entire sentence.
// Use "sequence_outputs" for token-level output.
let output_layer = bertModel.PooledOutput

let hidden_size = output_layer.shape |> Seq.last


let output_weights = tf.get_variable("output_weights6", 
                                     TensorShape([|hidden_size; NUM_LABELS|]), 
                                     initializer=tf.truncated_normal_initializer(stddev=0.02f))

let output_bias = tf.get_variable("output_bias6", 
                                  TensorShape(NUM_LABELS), 
                                  initializer=tf.zeros_initializer)

let (loss, predicted_labels, log_probs) =
    Tensorflow.Binding.tf_with(tf.variable_scope("loss"), fun _ -> 
        // Dropout helps prevent overfitting
        let output_layer = tf.nn.dropout(output_layer, keep_prob=tf.constant(0.9f))
        let logits = tf.matmul(output_layer, output_weights._AsTensor())
        let logits = tf.nn.bias_add(logits, output_bias)
        let log_probs = tf.log(tf.nn.softmax(logits, axis = -1))
        // Convert Labels into one-hot encoding
        let one_hot_labels = tf.one_hot(labels, depth=NUM_LABELS, dtype=tf.float32)
        let predicted_labels = tf.squeeze(tf.argmax(log_probs, axis = -1, output_type = tf.int32))
        /// If we're predicting, we want predicted labels and the probabiltiies.
        //if is_predicting:
        //  return (predicted_labels, log_probs)
        // If we're train/eval, compute loss between predicted and actual label
        let per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis= Nullable(-1))
        let loss = tf.reduce_mean(per_example_loss)
        (loss, predicted_labels, log_probs)
        )


// Optimization.create_optimizer()
//
//# We'll set sequences to be at most 128 tokens long.
//MAX_SEQ_LENGTH = 128
//# Convert our train and test features to InputFeatures that BERT understands.

let num_train_steps = int(float32 train.Length / float32 BATCH_SIZE * NUM_TRAIN_EPOCHS)
let num_warmup_steps = int(float32 num_train_steps * WARMUP_PROPORTION)

let train_op = Optimization.create_optimizer(loss, LEARNING_RATE, num_train_steps, Some(num_warmup_steps))

// TODO figure out dropout and how to turn it off easily....

let restore = tf.restore(Path.Combine(chkpt,"bert_model.ckpt"))

let sess = tf.Session()
sess.run(restore)

// TODO initialization

let subsample = train |> Array.subSample BATCH_SIZE 
let t1 = NDArray(subsample |> Array.map (fun x -> x.input_ids))
let t2 = NDArray(subsample |> Array.map (fun x -> x.input_mask))

sess.run(train_op, [|FeedItem(input_ids,t1); FeedItem(input_mask,t2)|])

ops.RegisterGradientFunction("Sqrt",Func<Operation,Tensor[],Tensor[]>(fun op grads -> 
    let grad = grads.[0]
    let y = op.outputs.[0]
    [|gen_ops.sqrt_grad(y,grad)|]))

ops.RegisterGradientFunction("Rsqrt",Func<Operation,Tensor[],Tensor[]>(fun op grads -> 
    let grad = grads.[0]
    let y = op.outputs.[0]
    [|gen_ops.rsqrt_grad(y,grad)|]))

/// Returns the gradient for (x-y)^2.
ops.RegisterGradientFunction("SquaredDifference",Func<Operation,Tensor[],Tensor[]>(fun op grads ->
    // TODO support skip_input_indices
    // TODO suport IndexedSlices
    let x = op.inputs.[0]
    let y = op.inputs.[1]
    let grad = grads.[0]
    let x_grad = 
        Tensorflow.Binding.tf_with(ops.control_dependencies([|grad|]), fun _ -> 
            2.0 *  grad * (x - y)
        )
    [|x_grad; -x_grad|]))


//            return tf_with(ops.control_dependencies(grads), delegate
//            {
//                y = math_ops.conj(y);
//                var factor = constant_op.constant(0.5f, dtype: y.dtype);
//                return new Tensor[] { grad * (factor * math_ops.reciprocal(y)) };
//            });
//)



//    let subsample = examples.[0..1]
//    let t1 = NDArray(subsample |> Array.map (fun x -> x.input_ids))
//    let t2 = NDArray(subsample |> Array.map (fun x -> x.input_mask))

//    let expected = [|-0.791169226f; -0.372503042f; -0.784386933f; 0.597510815f |]
//    let res = sess.run(bertModel.PooledOutput,[|FeedItem(input_ids,t1); FeedItem(input_mask,t2)|])
//                  .Data<float32>().ToArray().[0..3]



