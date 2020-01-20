[<AutoOpen>]
module Utils

open System.Reflection

// TODO - not sure if this should be open everywhere
//let tf = Tensorflow.Binding.tf

type Tensorflow.variable_scope with
    member this.name = 
        let m = typeof<Tensorflow.variable_scope>.GetField("_name", BindingFlags.Instance ||| BindingFlags.NonPublic)
        m.GetValue(this)  :?> string

let loggingf = printfn

type Tensorflow.tensorflow.train_internal with
    member this.get_or_create_global_step() : Tensorflow.RefVariable = 
        failwith "todo"


// https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/training/checkpoint_utils.py#L203-L291

// Replaces `tf.Variable` initializers so they load from a checkpoint file.
type Tensorflow.tensorflow.train_internal with
    member this.init_from_checkpoint(ckpt_dir_or_file, assignemnt_map) = 
        failwith "todo"
