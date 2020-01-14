// Apache 2.0 from https://github.com/google-research/bert/blob/e13c1f3459cc254f7abbabfc5a286a3304d573e4/run_pretraining.py
/// Run masked LM/next sentence masked_lm pre-training for BERT.
module RunPretraining

open Argu
open System
open System.IO
open Modeling
open Optimization
open Tensorflow

type Arguements =
    | [<Mandatory>] Bert_Config_File of path:string
    | [<Mandatory>] Input_File of path:string
    | [<Mandatory>] Output_Dir of dir:string
    | Init_Checkpoint of string
    | Max_Seq_Length of int
    | Max_Predictions_Per_Seq of int
    | Do_Train of bool
    | Do_Eval of bool
    | Train_Batch_Size of int
    | Eval_Batch_Size of int
    | Learning_Rate of float32
    | Num_Train_Steps of int
    | Num_Warmup_Steps of int
    | Save_Checkpoints_Steps of int
    | Iterations_Per_Loop of int
    | Max_Eval_Steps of int
    | Use_Tpu of bool
    | TPU_Name of string
    | TPU_Zone of string
    | GCP_Project of string option
    | Master of string option
    | Num_TPU_Cores of int option
    interface Argu.IArgParserTemplate with
        member this.Usage =
            match this with
            | Bert_Config_File _ -> 
                "The config json file corresponding to the pre-trained BERT model. " +
                "This specifies the model architecture."
            | Input_File _ -> 
                "Input TF example files (can be a glob or comma separated)."
            | Output_Dir _ -> 
                "The output directory where the model checkpoints will be written."
            | Init_Checkpoint _ -> 
                "Initial checkpoint (usually from a pre-trained BERT model)."
            | Max_Seq_Length _ -> 
                "The maximum total input sequence length after WordPiece tokenization. " +
                "Sequences longer than this will be truncated, and sequences shorter " + 
                "than this will be padded. Must match data generation."
            | Max_Predictions_Per_Seq _ -> 
                "Maximum number of masked LM predictions per sequence. " +
                "Must match data generation."
            | Do_Train _ -> "Whether to run training."
            | Do_Eval _ -> "Whether to run eval on the dev set."
            | Train_Batch_Size _ -> "Total batch size for training."
            | Eval_Batch_Size _ -> "Total batch size for eval."
            | Learning_Rate _ -> "The initial learning rate for Adam."
            | Num_Train_Steps _ -> "Number of training steps."
            | Num_Warmup_Steps _ -> "Number of warmup steps."
            | Save_Checkpoints_Steps _ -> "How often to save the model checkpoint."
            | Iterations_Per_Loop _ -> "How many steps to make in each estimator call."
            | Max_Eval_Steps _ -> "Maximum number of eval steps."
            | Use_Tpu _ -> "Whether to use TPU or GPU/CPU."
            | TPU_Name _ -> 
                "The Cloud TPU to use for training. This should be either the name " +
                "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url."
            | TPU_Zone _ -> 
                "[Optional] GCE zone where the Cloud TPU is located in. If not " +
                "specified, we will attempt to automatically detect the GCE project from " +
                "metadata."
            | GCP_Project _ -> 
                "[Optional] Project name for the Cloud TPU-enabled project. If not " + 
                "specified, we will attempt to automatically detect the GCE project from " +
                "metadata."
            | Master _ -> "[Optional] TensorFlow master URL."
            | Num_TPU_Cores _ -> "Only used if `use_tpu` is True. Total number of TPU cores to use."

// TODO defaults
// max_seq_length 128
// max_predictions_per_seq 20
// do_train false
// do_eval false
// train_batch_size 32
// eval_batch_size 8
// learning_rate 5e-5
// num_train_steps 100000
// num_warmup_steps 10000
// save_checkpoints_steps 1000
// iterations_per_loop 1000
// max_eval_steps 100
// use_tpu false
// num_tpu_cores

// TODO figure out tf.estimator.ModeKeys.TRAIN
type ModeKeys = 
    | Train
    | Eval

/// Gathers the vectors at the specific positions over a minibatch.
let gather_indexes(sequence_tensor, positions) = 
    let sequence_shape = Modeling.BertModel.get_shape_list(sequence_tensor, expected_rank=3)
    let batch_size = sequence_shape.[0]
    let seq_length = sequence_shape.[1]
    let width = sequence_shape.[2]
    let flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [|-1, 1|])
    let flat_positions = tf.reshape(positions + flat_offsets, [|-1|])
    let flat_sequence_tensor = tf.reshape(sequence_tensor, [|batch_size * seq_length; width|])
    let output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    output_tensor

let create_initializer(x:float32) : IInitializer = failwith "todo"
    
/// Get loss and log probs for the masked LM.
let get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights) = 
    let input_tensor = gather_indexes(input_tensor, positions)
    Binding.tf_with(tf.variable_scope("cls/predictions"), fun _ -> 
    // We apply one more non-linear transformation before the output layer.
    // This matrix is not used after pre-training.
        let input_tensor = 
            Binding.tf_with(tf.variable_scope("transform"), fun _ -> 
                let input_tensor = 
                    Modeling.Layers.dense(input_tensor,
                                    units=bert_config.hidden_size,
                                    activation = bert_config.hidden_act,
                                    kernel_initializer = create_initializer(bert_config.initializer_range))
                let input_tensor = Modeling.Layers.layer_norm(input_tensor)
                input_tensor)
        // The output weights are the same as the input embeddings, but there is
        // an output-only bias for each token.
        let output_bias = tf.get_variable("output_bias", 
                                          shape = TensorShape(bert_config.vocab_size.Value),
                                          initializer = tf.zeros_initializer)
        let logits = tf.matmul2(input_tensor, output_weights, transpose_b = true)
        let logits = tf.nn.bias_add(logits, output_bias)
        /// TODO the original tf.nn.log_softmax is more efficient
        let log_probs = tf.log(tf.nn.softmax(logits, axis = -1))

        let label_ids = tf.reshape(label_ids,[|-1|])
        let label_weights = tf.reshape(label_weights,[|-1|])

        let one_hot_labels = tf.one_hot(label_ids, 
                                        depth=bert_config.vocab_size.Value, 
                                        dtype=tf.float32)
        // The `positions` tensor might be zero-padded (if the sequence is too
        // short to have the maximum number of predictions). The `label_weights`
        // tensor has a value of 1.0 for every real prediction and 0.0 for the
        // padding predictions.
        let per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis= Nullable(-1))
        let numerator = tf.reduce_sum(label_weights * per_example_loss)
        let denominator = tf.reduce_sum(label_weights) + 1e-5
        let loss = numerator / denominator
        (loss, per_example_loss, log_probs))

/// Get loss and log probs for the next sentence prediction."""
let get_next_sentence_output(bert_config, input_tensor, labels) = 
    // Simple binary classification. Note that 0 is "next sentence" and 1 is
    // "random sentence". This weight matrix is not used after pre-training.
    let output_weights, output_bias = 
        Binding.tf_with(tf.variable_scope("cls/seq_relationship"), fun _ -> 
            let output_weights = 
                tf.get_variable("output_weights",
                                shape=TensorShape([|2; bert_config.hidden_size|]),
                                initializer= create_initializer(bert_config.initializer_range))
            let output_bias = 
                tf.get_variable("output_bias", 
                                shape= TensorShape([|2|]), 
                                initializer=tf.zeros_initializer)
            (output_weights, output_bias))
    let logits = tf.matmul2(input_tensor, output_weights._AsTensor(), transpose_b = true)
    let logits = tf.nn.bias_add(logits, output_bias)
    let log_probs = tf.log(tf.nn.softmax(logits, axis = -1))
    let labels = tf.reshape(labels, [|-1|])
    let one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    let per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis = Nullable(-1))
    let loss = tf.reduce_mean(per_example_loss)
    (loss, per_example_loss, log_probs)

/// Returns `model_fn` closure for TPUEstimator.
let model_fn_builder(bert_config : BertConfig, init_checkpoint : string, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings) = 
    /// The `model_fn` for TPUEstimator.
    let model_fn(features : Map<string,Tensor>, labels, mode : ModeKeys, params') = 
        loggingf "*** Features ***"
        for KeyValue(name,value) in features do
            loggingf "  name = %s, shape = %A"  name value.shape
        let input_ids = features.["input_ids"]
        let input_mask = features.["input_mask"]
        let segment_ids = features.["segment_ids"]
        let masked_lm_positions = features.["masked_lm_positions"]
        let masked_lm_ids = features.["masked_lm_ids"]
        let masked_lm_weights = features.["masked_lm_weights"]
        let next_sentence_labels = features.["next_sentence_labels"]
        let is_training = mode = ModeKeys.Train
        let model : BertModel = Modeling.BertModel(config = bert_config,
                                      is_training = is_training,
                                      input_ids = input_ids,
                                      input_mask = input_mask,
                                      token_type_ids = segment_ids,
                                      use_one_hot_embeddings = use_one_hot_embeddings)

        let (masked_lm_loss, maked_lm_example_loss, masked_lm_log_probs) = 
            get_masked_lm_output(bert_config, model.SequenceOutput, model.EmbeddingTable,
                                 masked_lm_positions, masked_lm_ids, masked_lm_weights)
        let (next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs) = 
            get_next_sentence_output(bert_config, model.PooledOutput, next_sentence_labels)

        let total_loss = masked_lm_loss + next_sentence_loss

        let tvars = tf.trainable_variables() |> Array.map (fun x -> x :?> RefVariable)

        let scaffold_fn, initialized_variable_names = 
            if not(String.IsNullOrWhiteSpace(init_checkpoint)) then
              let (assignment_map, initialized_variable_names) = 
                  Modeling.BertModel.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
              if use_tpu then
                let tpu_scaffold() =
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    // TODO tf.train.Scaffold()
                    failwith "todo"
                Some(tpu_scaffold), initialized_variable_names
              else 
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                None, initialized_variable_names
            else None, Map.empty

        loggingf "**** Trainable Variables ****"

        for var in tvars do
            let init_string = 
                if initialized_variable_names.ContainsKey(var.name) 
                then ", *INIT_FROM_CKPT*" 
                else ""
            loggingf "  name = %s, shape = %O%s" var.name var.shape init_string

////    output_spec = None
//    if mode = ModeKeys.Train
//    then
//        Optimization.create_optimizer(total_loss, 
//                                      learning_rate, 
//                                      num_train_steps, 
//                                      num_warmup_steps, 
//                                      use_tpu)
//
//        // TODO - not sure what to do here
////        let output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
////                                                          loss=total_loss,
////                                                          train_op=train_op,
////                                                          scaffold_fn=scaffold_fn)
//    elif mode = ModeKeys.Eval then
//        /// Computes the loss and accuracy of the model.
//        let metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
//                      masked_lm_weights, next_sentence_example_loss,
//                      next_sentence_log_probs, next_sentence_labels) =
//            let masked_lm_log_probs = 
//                tf.reshape(masked_lm_log_probs, [|-1; masked_lm_log_probs.shape |> Array.last|])
//            let masked_lm_predictions = 
//                tf.argmax(masked_lm_log_probs, axis = -1, output_type = tf.int32)
//            let masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [|-1|])
//            let masked_lm_ids = tf.reshape(masked_lm_ids, [|-1|])
//            let masked_lm_weights = tf.reshape(masked_lm_weights, [|-1|])
//            let masked_lm_accuracy = 
//                tf.metrics.accuracy(labels=masked_lm_ids,
//                                    predictions=masked_lm_predictions,
//                                    weights=masked_lm_weights)
//            let masked_lm_mean_loss = 
//                tf.metrics.mean(values=masked_lm_example_loss, weights=masked_lm_weights)
//            let next_sentence_log_probs = 
//                tf.reshape(next_sentence_log_probs, [|-1, next_sentence_log_probs.shape |> Array.last |])
//            let next_sentence_predictions = 
//                tf.argmax( next_sentence_log_probs, axis= -1, output_type=tf.int32)
//            let next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
//            let next_sentence_accuracy = 
//                tf.metrics.accuracy(labels=next_sentence_labels, predictions=next_sentence_predictions)
//            let next_sentence_mean_loss = tf.metrics.mean( values=next_sentence_example_loss)
//
//            [ "masked_lm_accuracy", masked_lm_accuracy
//            "masked_lm_loss", masked_lm_mean_loss
//            "next_sentence_accuracy", next_sentence_accuracy
//            "next_sentence_loss", next_sentence_mean_loss] |> Map.ofList
//
//        let eval_metrics = (metric_fn, [|
//              masked_lm_example_loss; masked_lm_log_probs; masked_lm_ids;
//              masked_lm_weights; next_sentence_example_loss;
//              next_sentence_log_probs; next_sentence_labels
//          |])
//
//        let output_spec = 
//            tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
//                                            loss=total_loss,
//                                            eval_metrics=eval_metrics,
//                                            scaffold_fn=scaffold_fn)
//        failwith "todo"
//    else
//        raise (ValueError(sprintf "Only TRAIN and EVAL modes are supported: %A"  (mode)))
//    //return output_spec
//  //return model_fn
//
//    model_fn
//
    failwith "todo"

//def input_fn_builder(input_files,
//                     max_seq_length,
//                     max_predictions_per_seq,
//                     is_training,
//                     num_cpu_threads=4):
//  """Creates an `input_fn` closure to be passed to TPUEstimator."""
//
//  def input_fn(params):
//    """The actual input function."""
//    batch_size = params["batch_size"]
//
//    name_to_features = {
//        "input_ids":
//            tf.FixedLenFeature([max_seq_length], tf.int64),
//        "input_mask":
//            tf.FixedLenFeature([max_seq_length], tf.int64),
//        "segment_ids":
//            tf.FixedLenFeature([max_seq_length], tf.int64),
//        "masked_lm_positions":
//            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
//        "masked_lm_ids":
//            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
//        "masked_lm_weights":
//            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
//        "next_sentence_labels":
//            tf.FixedLenFeature([1], tf.int64),
//    }
//
//    # For training, we want a lot of parallel reading and shuffling.
//    # For eval, we want no shuffling and parallel reading doesn't matter.
//    if is_training:
//      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
//      d = d.repeat()
//      d = d.shuffle(buffer_size=len(input_files))
//
//      # `cycle_length` is the number of parallel files that get read.
//      cycle_length = min(num_cpu_threads, len(input_files))
//
//      # `sloppy` mode means that the interleaving is not exact. This adds
//      # even more randomness to the training pipeline.
//      d = d.apply(
//          tf.contrib.data.parallel_interleave(
//              tf.data.TFRecordDataset,
//              sloppy=is_training,
//              cycle_length=cycle_length))
//      d = d.shuffle(buffer_size=100)
//    else:
//      d = tf.data.TFRecordDataset(input_files)
//      # Since we evaluate for a fixed number of steps we don't want to encounter
//      # out-of-range exceptions.
//      d = d.repeat()
//
//    # We must `drop_remainder` on training because the TPU requires fixed
//    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
//    # and we *don't* want to drop the remainder, otherwise we wont cover
//    # every sample.
//    d = d.apply(
//        tf.contrib.data.map_and_batch(
//            lambda record: _decode_record(record, name_to_features),
//            batch_size=batch_size,
//            num_parallel_batches=num_cpu_threads,
//            drop_remainder=True))
//    return d
//
//  return input_fn
//
//
//def _decode_record(record, name_to_features):
//  """Decodes a record to a TensorFlow example."""
//  example = tf.parse_single_example(record, name_to_features)
//
//  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
//  # So cast all int64 to int32.
//  for name in list(example.keys()):
//    t = example[name]
//    if t.dtype == tf.int64:
//      t = tf.to_int32(t)
//    example[name] = t
//
//  return example

//def main(_):
//  tf.logging.set_verbosity(tf.logging.INFO)
//
//  if not FLAGS.do_train and not FLAGS.do_eval:
//    raise ValueError("At least one of `do_train` or `do_eval` must be True.")
//
//  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
//
//  tf.gfile.MakeDirs(FLAGS.output_dir)
//
//  input_files = []
//  for input_pattern in FLAGS.input_file.split(","):
//    input_files.extend(tf.gfile.Glob(input_pattern))
//
//  tf.logging.info("*** Input Files ***")
//  for input_file in input_files:
//    tf.logging.info("  %s" % input_file)
//
//  tpu_cluster_resolver = None
//  if FLAGS.use_tpu and FLAGS.tpu_name:
//    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
//        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
//
//  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
//  run_config = tf.contrib.tpu.RunConfig(
//      cluster=tpu_cluster_resolver,
//      master=FLAGS.master,
//      model_dir=FLAGS.output_dir,
//      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
//      tpu_config=tf.contrib.tpu.TPUConfig(
//          iterations_per_loop=FLAGS.iterations_per_loop,
//          num_shards=FLAGS.num_tpu_cores,
//          per_host_input_for_training=is_per_host))
//
//  model_fn = model_fn_builder(
//      bert_config=bert_config,
//      init_checkpoint=FLAGS.init_checkpoint,
//      learning_rate=FLAGS.learning_rate,
//      num_train_steps=FLAGS.num_train_steps,
//      num_warmup_steps=FLAGS.num_warmup_steps,
//      use_tpu=FLAGS.use_tpu,
//      use_one_hot_embeddings=FLAGS.use_tpu)
//
//  # If TPU is not available, this will fall back to normal Estimator on CPU
//  # or GPU.
//  estimator = tf.contrib.tpu.TPUEstimator(
//      use_tpu=FLAGS.use_tpu,
//      model_fn=model_fn,
//      config=run_config,
//      train_batch_size=FLAGS.train_batch_size,
//      eval_batch_size=FLAGS.eval_batch_size)
//
//  if FLAGS.do_train:
//    tf.logging.info("***** Running training *****")
//    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
//    train_input_fn = input_fn_builder(
//        input_files=input_files,
//        max_seq_length=FLAGS.max_seq_length,
//        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
//        is_training=True)
//    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
//
//  if FLAGS.do_eval:
//    tf.logging.info("***** Running evaluation *****")
//    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
//
//    eval_input_fn = input_fn_builder(
//        input_files=input_files,
//        max_seq_length=FLAGS.max_seq_length,
//        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
//        is_training=False)
//
//    result = estimator.evaluate(
//        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)
//
//    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
//    with tf.gfile.GFile(output_eval_file, "w") as writer:
//      tf.logging.info("***** Eval results *****")
//      for key in sorted(result.keys()):
//        tf.logging.info("  %s = %s", key, str(result[key]))
//        writer.write("%s = %s\n" % (key, str(result[key])))
//
//
//if __name__ == "__main__":
//  flags.mark_flag_as_required("input_file")
//  flags.mark_flag_as_required("bert_config_file")
//  flags.mark_flag_as_required("output_dir")
//  tf.app.run()
