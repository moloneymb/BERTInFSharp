// Apache 2.0 https://github.com/google-research/bert/blob/master/run_classifier.py
module RunClassifier

module tf = 
    module logging =
        let info(text : string) = failwith "todo"

open Argu

type Arguments =
    | [<Mandatory>] Data_Dir of path:string
    | [<Mandatory>] Bert_Config_File of path:string
    | [<Mandatory>] Task_Name of string
    | [<Mandatory>] Vocab_File of path:string
    | [<Mandatory>] Output_Dir of path:string
    | Init_Checkpoint of path:string option
    | Do_Lower_Case of bool option
    | Max_Seq_Length of int option
    | Do_Train of bool option
    | Do_Eval of bool option
    | Do_Predict of bool option
    | Train_Batch_Size of int option
    | Eval_Batch_Size of int option
    | Predict_Batch_Size of int option
    | Learning_Rate of float option
    | Num_Train_Epochs of float option
    | Warmup_Proportion of float option
    | Save_Checkpoints_Steps of int option
    | Iterations_Per_Loop of int option
    | Use_TPU of bool option
    | TPU_Name of string option
    | TPU_Zone of string option
    | GCP_Project of string option
    | Master of string option
    | Num_TPU_Cores of int option
    interface Argu.IArgParserTemplate with
        member this.Usage =
            match this with
            | Data_Dir _ -> "The input data dir. Should contain the .tsv files (or other data files) for the task."
            | Bert_Config_File _ -> "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture."
            | Task_Name _ -> "The name of the task to train."
            | Vocab_File _ -> "The vocabulary file that the BERT model was trained on."
            | Output_Dir _ -> "The output directory where the model checkpoints will be written."
            | Init_Checkpoint _ -> "Initial checkpoint (usually from a pre-trained BERT model)."
            | Do_Lower_Case _ -> "Whether to lower case the input text. Should be True for uncased models and False for cased models."
            | Max_Seq_Length _ -> 
                "The maximum total input sequence length after WordPiece tokenization. " + 
                "Sequences longer than this will be truncated, and sequences shorter " + 
                "than this will be padded."
            | Do_Train _ -> "Whether to run training."
            | Do_Eval _ -> "Whether to run eval on the dev set."
            | Do_Predict _ -> "Whether to run the model in inference mode on the test set."
            | Train_Batch_Size _ -> "Total batch size for training."
            | Eval_Batch_Size _ -> "Total batch size for eval."
            | Predict_Batch_Size _ -> "Total batch size for predict."
            | Learning_Rate _ -> "The initial learning rate for Adam."
            | Num_Train_Epochs _ -> "Total number of training epochs to perform."
            | Warmup_Proportion _ -> "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."
            | Save_Checkpoints_Steps _ -> "How often to save the model checkpoint."
            | Iterations_Per_Loop _ -> "How many steps to make in each estimator call."
            | Use_TPU _ -> "Whether to use TPU or GPU/CPU."
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

// TODO Defaults for options
// do_lower_case true
// max_seq_length 128
// do_train false
// do_eval false
// do_predict false
// train_batch_size 32
// eval_batch_size 8
// predict_batch_size 8
// learning_rate 5e-5
// num_train_epochs 3.0
// warmup_proportion 0.1
// save_checkpoints_steps 1000
// iterations_per_loop 1000
// use_tpu false
// num_tpu_cores 8

// <summary>A single training/test example for simple sequence classification </summary>
// <param name="guid">Unique id for the example.</param>
// <param name="text_a"> string. The untokenized text of the first sequence. For single
// sequence tasks, only this sequence must be specified.</param>
// <param name="text_b"> (Optional) string. The untokenized text of the second sequence.
//  Only must be specified for sequence pair tasks.</param>
// <param name="label"> (Optional) string. The label of the example. This should be
// specified for train and dev examples, but not for test examples </param>
type InputExample(guid : System.Guid, text_a : string, ?text_b : string, ?label : string) = 
    member this.guid   = guid
    member this.text_a = text_a
    member this.text_b = text_b
    member this.label  = label


// Fake example so the num input examples is a multiple of the batch size.
//  When running eval/predict on the TPU, we need to pad the number of examples
//  to be a multiple of the batch size, because the TPU requires a fixed batch
//  size. The alternative is to drop the last batch, which is bad because it means
//  the entire output data won't be generated.
//  We use this class instead of `None` because treating `None` as padding
//  battches could cause silent errors.
type PaddingInputExample() = 
    class
    end

/// A single set of features of data.
type InputFeatures(input_ids,
                   input_mask,
                   segment_ids,
                   label_id,
                   ?is_real_example) =
    member this.input_ids = input_ids
    member this.input_mask = input_mask
    member this.segment_ids = segment_ids
    member this.label_id = label_id
    member this.is_real_example = defaultArg is_real_example false

/// NOTE: Should I get rid of the default methods
/// Base class for data converters for sequence classification data sets.
type DataProcessor() = 
    abstract member get_train_examples : string -> string
    /// Gets a collection of `InputExample`s for the train set.
    default this.get_train_examples(data_dir) = raise (System.NotImplementedException())

    abstract member get_dev_examples : string -> string
    /// Gets a collection of `InputExample`s for the dev set.
    default this.get_dev_examples(data_dir) = raise (System.NotImplementedException())

    abstract member get_test_examples : string -> string
    /// Gets a collection of `InputExample`s for prediction.
    default this.get_test_examples (data_dir) = raise (System.NotImplementedException())

    abstract member get_labels : unit -> string []
    /// Gets the list of labels for this data set."""
    default this.get_labels () = raise (System.NotImplementedException())

    /// Reads a tab separated value file.
    static member private read_tsv(input_file : string, ?quotechar : char)  =
        let reader = new System.IO.StreamReader(input_file)
        let config = CsvHelper.Configuration.Configuration(Delimiter = "\t", Quote = (defaultArg quotechar '"'))
        use csv = new CsvHelper.CsvReader(reader, config)
        if not(csv.Read()) then [||]
        else 
            // NOTE this is a hack because csv.Context.ColumnCount always returned 0 in testing
            // and there seemed to be no other way to get the column count
            let rec getColumns(col : int) = 
                try 
                    csv.[col] |> ignore
                    getColumns(col+1)
                with
                | :? CsvHelper.MissingFieldException -> col
            let colCount = getColumns(0)
            let getRow() = [|for i in 0 .. colCount - 1 -> csv.[i]|]
            [|
                yield getRow()
                while csv.Read() do
                    yield getRow()
            |]


// run_classifier.InputExample
// run_classifier.convert_examples_to_features
// run_classifier.convert_single_to_features
// run_classifier.input_fn_builder

let convert_single_example (ex_index, example, label_list, max_seq_length, tokenizer) = failwith "todo"

// NOTE: This function is not used by this file but is still used by the Colab and
// people who depend on it.
/// Convert a set of `InputExample`s to a list of `InputFeatures`.
let convert_examples_to_features(examples : string[], label_list, max_seq_length : int, tokenizer : string -> string[]) =
    examples |> Array.iteri (fun ex_index example -> 
        if ex_index % 10000 = 0 then
            tf.logging.info(sprintf "Writing example %d of %d" ex_index examples.Length)
        convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer))


