open System
open Modeling
open Tensorflow

//type BertModelTester() = 
//    /// Creates a random int32 tensor of the shape within the vocab size.
//    static member ids_tensor(shape: int[], vocab_size: int, ?rng: Random, ?name: string) = 
//        let rng = defaultArg rng (Random())
//        let total_dims = shape |> Array.fold (fun x y -> x*y) 1
//        let data = [|for i in 0 .. total_dims - 1 -> rng.Next(vocab_size)|]
//        match name with
//        | Some(name) -> tf.constant(data, dtype=tf.int32,shape=shape,name=name)
//        | _ -> tf.constant(data, dtype=tf.int32,shape=shape)

[<EntryPoint>]
let main argv =
//    let batch_size = 2 
//    let seq_length = 7
//    let vocab_size = 99
//    let input_ids = BertModelTester.ids_tensor([|batch_size; seq_length|], vocab_size)
//    let config = {BertConfig.Default with vocab_size = Some(vocab_size)}
//    let bertModel = BertModel(config, true, input_ids)
//    tf.get_default_graph().get_operations() |> Seq.filter (fun x -> x.op.OpType = "VariableV2") |> Seq.iter (fun x -> printfn "%s" x.name)
    0 

//
//            var params_shape = array_ops.shape(@params, out_type: tf.int64);
//            params_shape = math_ops.cast(params_shape, tf.int32);
//
//            var indices = op.inputs[1];
//            var indices_size = array_ops.expand_dims(array_ops.size(indices), 0);
//            var axis = op.inputs[2];
//            var axis_static = tensor_util.constant_value(axis);
//
//            // For axis 0 gathers, build an appropriately shaped IndexedSlices.
//            if((int)axis_static == 0)
//            {
//                var params_tail_shape = params_shape.slice(new NumSharp.Slice(start:1));
//                var values_shape = array_ops.concat(new[] { indices_size, params_tail_shape }, 0);
//                var values = array_ops.reshape(grad, values_shape);
//                indices = array_ops.reshape(indices, indices_size);
//                return new Tensor[]
//                {
//                    new IndexedSlices(values, indices, params_shape),
//                    null,
//                    null
//                };
//            }

//[<EntryPoint>]
//let main argv =
//    let vocab_size = 13
//    let embedding_size = 35
//    let input_ids = BertModelTester.ids_tensor([|7|], vocab_size)
//    let embedding_table = 
//        tf.get_variable(
//                        name="word_embeddings", 
//                        shape=Tensorflow.TensorShape(vocab_size, embedding_size),
//                        initializer=Utils.create_initializer(0.02f)
//                        )._AsTensor()
//
//    let output = gen_ops.gather_v2(embedding_table, input_ids, tf.constant(0))
//    let grad = tf.gradients(output, embedding_table).[0]
//
//    printfn "InputID shape %A" input_ids.shape
//    printfn "TVar %A %s" embedding_table.shape embedding_table.name
//    printfn "Grad %A %s" grad.shape grad.name
//    printfn "%A" output.shape
//    use sess = tf.Session()
//    let init = tf.global_variables_initializer()
//    sess.run(init)
//    let res = sess.run(grad)
//    printfn "%A" res
//    0 



