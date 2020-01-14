/// The main BERT model and related functions.
module Modeling
// Apache License, Version 2.0
// Converted to F# from https://github.com/google-research/bert/blob/cc7051dc592802f501e8a6f71f8fb3cf9de95dc9/modeling.py


/// NOTE: mistakes in BERT L350 s/dimension/value
// TODO tensor alias
// TODO properly manage GraphKeys
// TODO proper matmul

// TODO improvements
// np.sqrt(2.0 / np.pi) needs a 
// Tensor[] -> TensorShape needed to be able to build a function of dynamic shapes

#if INTERACTIVE
#I @"C:\Users\moloneymb\.nuget\packages\"
#r @"numsharp\0.20.4\lib\netstandard2.0\NumSharp.Core.dll"
#r @"tensorflow.net\0.13.0\lib\netstandard2.0\TensorFlow.NET.dll"
#r @"system.memory\4.5.3\lib\netstandard2.0\System.Memory.dll"
#r @"google.protobuf\3.10.1\lib\netstandard2.0\Google.Protobuf.dll"
#endif

open NumSharp
open System
open System.Linq
open Tensorflow
open Tensorflow.Operations.Activation


type gen_ops = Tensorflow.Operations.gen_ops

[<AutoOpen>]
module Auto = 
    //https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/ops/math_ops.py#L2565-L2754
    type Tensorflow.tensorflow with
        //<summary>Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
        //  The inputs must, following any transpositions, be tensors of rank >= 2
        //  where the inner 2 dimensions specify valid matrix multiplication arguments,
        //  and any further outer dimensions match.
        //  Both matrices must be of the same type. The supported types are:
        //  `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.
        //  Either matrix can be transposed or adjointed (conjugated and transposed) on
        //  the fly by setting one of the corresponding flag to `True`. These are `False`
        //  by default.
        //  If one or both of the matrices contain a lot of zeros, a more efficient
        //  multiplication algorithm can be used by setting the corresponding
        //  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
        //  This optimization is only available for plain matrices (rank-2 tensors) with
        //  datatypes `bfloat16` or `float32`.
        //  For example:
        //  ```python
        //  # 2-D tensor `a`
        //  # [[1, 2, 3],
        //  #  [4, 5, 6]]
        //  a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
        //  # 2-D tensor `b`
        //  # [[ 7,  8],
        //  #  [ 9, 10],
        //  #  [11, 12]]
        //  b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
        //  # `a` * `b`
        //  # [[ 58,  64],
        //  #  [139, 154]]
        //  c = tf.matmul(a, b)
        //  # 3-D tensor `a`
        //  # [[[ 1,  2,  3],
        //  #   [ 4,  5,  6]],
        //  #  [[ 7,  8,  9],
        //  #   [10, 11, 12]]]
        //  a = tf.constant(np.arange(1, 13, dtype=np.int32),
        //                  shape=[2, 2, 3])
        //  # 3-D tensor `b`
        //  # [[[13, 14],
        //  #   [15, 16],
        //  #   [17, 18]],
        //  #  [[19, 20],
        //  #   [21, 22],
        //  #   [23, 24]]]
        //  b = tf.constant(np.arange(13, 25, dtype=np.int32),
        //                  shape=[2, 3, 2])
        //  # `a` * `b`
        //  # [[[ 94, 100],
        //  #   [229, 244]],
        //  #  [[508, 532],
        //  #   [697, 730]]]
        //  c = tf.matmul(a, b)
        //  # Since python >= 3.5 the @ operator is supported (see PEP 465).
        //  # In TensorFlow, it simply calls the `tf.matmul()` function, so the
        //  # following lines are equivalent:
        //  d = a @ b @ [[10.], [11.]]
        //  d = tf.matmul(tf.matmul(a, b), [[10.], [11.]])
        //  ```
        // </summary>
        // <param name="a"> `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
        // `complex128` and rank > 1. </param>
        // <param name="b"> `Tensor` with same type and rank as `a`.</param>
        // <param name="transpose_a"> If `True`, `a` is transposed before multiplication.</param>
        // <param name="transpose_b"> If `True`, `b` is transposed before multiplication.</param>
        // <param name="adjoint_a"> If `True`, `a` is conjugated and transposed before
        // multiplication.</param>
        // <param name="adjoint_b"> If `True`, `b` is conjugated and transposed before multiplication. </param>
        // <param name="a_is_sparse"> If `True`, `a` is treated as a sparse matrix.</param>
        // <param name="b_is_sparse"> If `True`, `b` is treated as a sparse matrix.</param>
        // <param name="name"> Name for the operation (optional).</param>
        // <returns>
        //    A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
        //    the product of the corresponding matrices in `a` and `b`, e.g. if all
        //    transpose or adjoint attributes are `False`:
        //    `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
        //    for all indices i, j.
        //    Note: This is matrix product, not element-wise product.
        // </returns>
        // <exception cdef="Tensorflow.ValueError">
        // If transpose_a and adjoint_a, or transpose_b and adjoint_b
        // are both set to True.
        // </exception>
        member this.matmul2(a : Tensor, 
                             b : Tensor, 
                             ?transpose_a : bool,
                             ?transpose_b : bool,
                             ?adjoint_a : bool,
                             ?adjoint_b : bool,
                             ?a_is_sparse : bool,
                             ?b_is_sparse : bool,
                             ?name : string) = 
            let tf = Tensorflow.Binding.tf
            let transpose_a = defaultArg transpose_a false
            let transpose_b = defaultArg transpose_b false
            let adjoint_a = defaultArg adjoint_a false
            let adjoint_b = defaultArg adjoint_b false
            let a_is_sparse = defaultArg a_is_sparse false
            let b_is_sparse = defaultArg b_is_sparse false
            let name = defaultArg name String.Empty
            Tensorflow.Binding.tf_with(tf.name_scope(name,"MatMul",[|a;b|]), fun (name:ops.NameScope) -> 
                if transpose_a && adjoint_a then
                    raise (ValueError("Only one of transpose_a and adjoint_a can be True."))
                if transpose_b && adjoint_b then
                    raise (ValueError("Only one of transpose_b and adjoint_b can be True."))
                if a_is_sparse || b_is_sparse then failwith "todo"
                if adjoint_a || adjoint_b then failwith "todo"
    //            let output_may_have_non_empty_batch_shape, batch_mat_mul_fn =
    //                true, tf.matmul
      //          if false then
                let output_may_have_non_empty_batch_size = (a.shape.Length > 2) && (b.shape.Length > 2)
                if (not a_is_sparse) && (not b_is_sparse) && output_may_have_non_empty_batch_size then
                    // BatchMatmul does not support transpose, so we conjugate the matrix and
                    // use adjoint instead. Conj() is a noop for real matrices.
                    let conj = id
                    //https://github.com/tensorflow/tensorflow/blob/590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b/tensorflow/python/ops/math_ops.py#L3416
                    if a.dtype.is_complex() || b.dtype.is_complex() then failwith "todo"
                    let a, adjoint_a = if transpose_a then conj(a), true else a,adjoint_a
                    let b, adjoint_b = if transpose_b then conj(b), true else b,adjoint_b
                    Tensorflow.gen_math_ops.batch_mat_mul(a, b, adj_x=adjoint_a, adj_y=adjoint_b, name=name._name)
                else
                    if a.shape.Length = 2 && b.shape.Length = 2 && not(transpose_a) && not(transpose_b) then tf.matmul(a,b)
                    else failwith "todo"
                )

let tf = Tensorflow.Binding.tf

module utils = 
//https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/layers/python/layers/utils.py
//    def get_variable_collections(variables_collections, name):
//      if isinstance(variables_collections, dict):
//        variable_collections = variables_collections.get(name, None)
//      else:
//        variable_collections = variables_collections
//      return variable_collections
//    let get_variable_collections(variables_collections : string[], name : string) : string[] = 
//        variables_collection
//
//    let get_variable_collections(variables_collections : Map<string,string[]>, name : string) : string[] = 
//        variable_collections.[name]

      ///<summary>Append an alias to the list of aliases of the tensor.</summary>
      ///<param name=tensor>A `Tensor`</param>
      ///<param name=alias>String, to add to the list of aliases of the tensor.</param>
      ///<returns> The tensor with a new alias appended to its list of aliases.</returns>
    let append_tensor_alias(tensor : Tensor, alias : string) = 
        let dropSlash(x:string) = 
            if x.[x.Length-1] = '/' 
            then if x = "/" then "" else x.[.. x.Length - 2] 
            else x

        let alias = dropSlash(alias)
////    TODO - tensors do not have alias yet. We're ignoring this for now
//          if hasattr(tensor, 'aliases'):
//            tensor.aliases.append(alias)
//          else:
//            tensor.aliases = [alias]
//          return tensor
        tensor

//https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/layers/python/layers/utils.py
    let collect_named_outputs(collections : string[], alias : string, outputs : Tensor) = 
        if collections.Length > 0 then 
            ops.add_to_collections(System.Collections.Generic.List<string>(collections), outputs)
        outputs
        
// SEE https://stackoverflow.com/questions/47608357/difference-between-get-variable-and-model-variable-function
//https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/contrib/framework/python/ops/variables.py
type variables with
    static member model_variable(name : string, 
                                 shape : TensorShape, 
                                 dtype : TF_DataType, 
                                 initializer : IInitializer,
                                 collections : string[],
                                 trainable : bool) = 

        let collections = System.Collections.Generic.List<string>(collections)
        collections.Add(ops.GraphKeys.GLOBAL_VARIABLES_)
        collections.Add(ops.GraphKeys.MODEL_VARIABLES_)

        let v = 
            tf.get_variable(name,
                            shape = shape,
                            dtype = dtype,
                            initializer = initializer,
                            collections = collections,
                            trainable = Nullable(trainable))
        v


type Layers () = 
    static member dense(input : Tensor, 
                        units : int, 
                        ?activation : IActivation, 
                        ?use_bias : bool,
                        ?kernel_initializer : IInitializer,
                        ?bias_initializer : IInitializer,
                        ?trainable : Nullable<bool>,
                        ?reuse : Nullable<bool>,
                        ?name : string) =
        let dtype = input.dtype
        let name = defaultArg name String.Empty
        let reuse = defaultArg reuse (Nullable<bool>())
        let trainable = defaultArg trainable (Nullable<bool>())
        let use_bias = defaultArg use_bias true
        let bias_initializer = defaultArg bias_initializer tf.zeros_initializer
        Tensorflow.Binding.tf_with(tf.name_scope(name,"dense",[|input|]), fun (ns:Tensorflow.ops.NameScope) ->
            Tensorflow.Binding.tf_with(tf.variable_scope(name,"dense",reuse=reuse), fun _vs ->
                match input.shape with
                | [|_;n|] when n > 0 ->
                    let kernel = tf.get_variable("kernel",TensorShape(n,units),dtype=dtype,?initializer=kernel_initializer,trainable=trainable)
                    let x = 
                        if use_bias 
                        then
                            let bias = tf.get_variable("bias",TensorShape(units),dtype=dtype,initializer=bias_initializer,trainable=trainable)
                            gen_ops.bias_add(gen_ops.mat_mul(input,kernel._AsTensor()),bias._AsTensor())
                        else
                            gen_ops.mat_mul(input,kernel._AsTensor())
                    let x = match activation with None -> x | Some(f) -> f.Activate(x)
                    tf.identity(x,name=Tensorflow.ops.NameScope.op_Implicit(ns))
                | _ ->
                    raise (ValueError(sprintf "Input shape of %A is not suitable for a dense network " input.shape))
            )
        )

    // https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/nn_impl.py#L1382-L1442
    ///<summary>Batch normalization.
    ///Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
    ///`scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):
    ///\\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)
    ///`mean`, `variance`, `offset` and `scale` are all expected to be of one of two
    ///shapes:
    ///* In all generality, they can have the same number of dimensions as the
    ///  input `x`, with identical sizes as `x` for the dimensions that are not
    ///  normalized over (the 'depth' dimension(s)), and dimension 1 for the
    ///  others which are being normalized over.
    ///  `mean` and `variance` in this case would typically be the outputs of
    ///  `tf.nn.moments(..., keep_dims=True)` during training, or running averages
    ///  thereof during inference.
    ///* In the common case where the 'depth' dimension is the last dimension in
    ///  the input tensor `x`, they may be one dimensional tensors of the same
    ///  size as the 'depth' dimension.
    ///  This is the case for example for the common `[batch, depth]` layout of
    ///  fully-connected layers, and `[batch, height, width, depth]` for
    ///  convolutions.
    ///  `mean` and `variance` in this case would typically be the outputs of
    ///  `tf.nn.moments(..., keep_dims=False)` during training, or running averages
    ///  thereof during inference.
    ///See Source: [Batch Normalization: Accelerating Deep Network Training by
    ///Reducing Internal Covariate Shift; S. Ioffe, C. Szegedy]
    ///(http://arxiv.org/abs/1502.03167).
    ///</summary>
    /// <param name="x">Input `Tensor` of arbitrary dimensionality.</param>
    /// <param name="mean"> A mean `Tensor`.</param>
    /// <param name="variance"> A variance `Tensor`.</param>
    /// <param name="offset"> An offset `Tensor`, often denoted \\(\beta\\) in equations, or
    ///  None. If present, will be added to the normalized tensor.
    ///     scale: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
    ///      `None`. If present, the scale is applied to the normalized tensor.</param>
    /// <param name="variance_epsilon"> A small float number to avoid dividing by 0.</param>
    /// <param name="name"> A name for this operation (optional).</param>
    /// <returns> the normalized, scaled, offset tensor.</returns>
    static member batch_normalization(x : Tensor,
                                      mean : Tensor,
                                      variance : Tensor,
                                      ?offset : Tensor,
                                      ?scale : Tensor,
                                      ?epsilon : float,
                                      ?name : string) =
        let epsilon = defaultArg epsilon 1.0e-6
        let inputs : Tensor[] = [| Some(x); Some(mean); Some(variance); scale; offset|] |> Array.choose id
        Tensorflow.Binding.tf_with(ops.name_scope(defaultArg name "", "batchnorm",inputs), fun ns ->
            let inv = math_ops.rsqrt(variance + epsilon)
            let inv2 = match scale with | Some(scale) -> inv * scale | _ -> inv
            //let rhs = math_ops.cast(inv, x.dtype) + math_ops.cast((match offset with | None -> -mean * inv | Some(offset) -> offset - mean * inv), x.dtype)
            let rhs1 = mean * inv2
            let rhs = offset.Value - rhs1
            x * rhs)

    // https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/layers/python/layers/layers.py#L2204
    /// <summary>
    ///Adds a Layer Normalization layer.
    ///  Based on the paper:
    ///    "Layer Normalization"
    ///    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    ///    https://arxiv.org/abs/1607.06450.
    ///  Can be used as a normalizer function for conv2d and fully_connected.
    ///  Given a tensor `inputs` of rank `R`, moments are calculated and normalization
    ///  is performed over axes `begin_norm_axis ... R - 1`.  Scaling and centering,
    ///  if requested, is performed over axes `begin_params_axis .. R - 1`.
    ///  By default, `begin_norm_axis = 1` and `begin_params_axis = -1`,
    ///  meaning that normalization is performed over all but the first axis
    ///  (the `HWC` if `inputs` is `NHWC`), while the `beta` and `gamma` trainable
    ///  parameters are calculated for the rightmost axis (the `C` if `inputs` is
    ///  `NHWC`).  Scaling and recentering is performed via broadcast of the
    ///  `beta` and `gamma` parameters with the normalized tensor.
    ///  The shapes of `beta` and `gamma` are `inputs.shape[begin_params_axis:]`,
    ///  and this part of the inputs' shape must be fully defined.
    /// </summary>
    /// <param name="inputs"> A tensor having rank `R`. The normalization is performed over axes
    ///  `begin_norm_axis ... R - 1` and centering and scaling parameters are
    ///  calculated over `begin_params_axis ... R - 1`.</param>
    /// <param name="center"> If True, add offset of `beta` to normalized tensor. If False, `beta`
    ///  is ignored. </param>
    /// <param name="scale"> If True, multiply by `gamma`. If False, `gamma` is not used. When the
    ///  next layer is linear (also e.g. `nn.relu`), this can be disabled since the
    ///  scaling can be done by the next layer. </param>
    /// <param name="activation_fn"> Activation function, default set to None to skip it and
    ///  maintain a linear activation. </param>
    /// <param name="reuse"> Whether or not the layer and its variables should be reused. To be
    ///  able to reuse the layer scope must be given.</param>
    /// <param name="variables_collections"> Optional collections for the variables.</param>
    /// <param name="outputs_collections"> Collections to add the outputs.</param>
    /// <param name="trainable"> If `True` also add variables to the graph collection
    ///   `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).</param>
    /// <param name="begin_norm_axis"> The first normalization dimension: normalization will be
    ///   performed along dimensions `begin_norm_axis : rank(inputs)` </param>
    /// <param name="begin_params_axis"> The first parameter (beta, gamma) dimension: scale and
    ///  centering parameters will have dimensions 
    ///  `begin_params_axis : rank(inputs)` and will be broadcast with the
    ///    normalized inputs accordingly. </param>
    /// <param name="scope"> Optional scope for `variable_scope`.</param>
    /// <returns>A `Tensor` representing the output of the operation, having the same
    /// shape and dtype as `inputs`.  </returns>
    /// <exception cdef="Tensorflow.ValueError">
    /// If the rank of `inputs` is not known at graph build time,
    /// or if `inputs.shape[begin_params_axis:]` is not fully defined at
    /// graph build time.  </exception>
    static member layer_norm(inputs : Tensor,
                             ?center : bool,
                             ?scale : bool,
                             ?activation_fn : IActivation,
                             ?reuse : bool, 
                             ?variables_collections : Map<string,string[]>, 
                             ?output_collections : string[], 
                             ?trainable : bool,
                             ?begin_norm_axis : int,
                             ?begin_params_axis : int,
                             ?scope : string) = 

        //let scope = defaultArg scope "LayerNorm"
        let center = defaultArg center true
        let scale = defaultArg scale true
        let trainable = defaultArg trainable true
        let begin_norm_axis = defaultArg begin_norm_axis 1
        let begin_params_axis = defaultArg begin_params_axis 1
        let variables_collections = defaultArg variables_collections (Map(["beta",[||];"gamma",[||]]))
        
        Tensorflow.Binding.tf_with(
            tf.variable_scope("LayerNorm", // TODO this is a hack
                              "LayerNorm", 
                              values = [|inputs|],
                              reuse = (reuse |> Option.toNullable)), fun (vs:variable_scope) ->

            //let inputs = ops.convert_to_tensor(inputs)
            // NOTE: TensorShape GetSlice is not defined which is needed for slicing
            let inputs_shape = inputs.TensorShape.dims
            let inputs_rank = inputs_shape.Length - 1
            if inputs_rank = 0 then
                raise (ValueError(sprintf "Inputs %s has undefined rank." inputs.name))
            let dtype = inputs.dtype.as_base_dtype()
            let begin_norm_axis = 
                if begin_norm_axis < 0 
                then inputs_rank + begin_norm_axis
                else begin_norm_axis
            if begin_params_axis >= inputs_rank || begin_norm_axis >= inputs_rank then
                raise (ValueError(sprintf "begin_params_axis (%d) and begin_norm_axis (%d) must be < rank(inputs) (%d)"
                                    begin_params_axis begin_norm_axis inputs_rank))

            //params_shape = inputs_shape[begin_params_axis:]
            let params_shape = 
                inputs_shape.[(if begin_params_axis < 0 then begin_params_axis + inputs_shape.Length else begin_params_axis) .. ] 
                |> TensorShape
                
            if not(params_shape.is_fully_defined()) then
                raise (ValueError(sprintf "Inputs %s: shape(inputs)[%i:] is not fully defined: %i"
                                    inputs.name begin_params_axis inputs_rank))
            // Allocate parameters for the beta and gamma of the normalization.
            let beta : Tensor option = 
                if center 
                then
                    //let beta_collections = utils.get_variable_collections(variables_collections,"beta")
                    let beta_collections = variables_collections.["beta"]
                    variables.model_variable("beta",
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=tf.zeros_initializer,
                                             collections = beta_collections,
                                             trainable = trainable )._AsTensor() |> Some
                else None
            let gamma : Tensor option = 
                if scale 
                then
                    //let gamma_collections = utils.get_variable_collections(variables_collections,"gamma")
                    let gamma_collections = variables_collections.["gamma"]
                    variables.model_variable("gamma",
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=tf.ones_initializer,
                                             collections = gamma_collections,
                                             trainable = trainable )._AsTensor() |> Some
                    
                else None
            // By default, compute the moments across all the dimensions except the one with index 0.
            let norm_axes = [|begin_norm_axis .. inputs_rank-1|]
            let (mean, variance) = tf.nn.moments(inputs, norm_axes, keep_dims = true).ToTuple()
            // Compute layer normalization using the batch_normalization function.
            // Note that epsilon must be increased for float16 due to the limited
            // representable range.
            let variance_epsilon = if dtype = dtypes.float16 then 1e-3 else 1e-12
            let outputs = Layers.batch_normalization(inputs, 
                                                    mean, 
                                                    variance, 
                                                    ?offset=beta, 
                                                    ?scale=gamma, 
                                                    epsilon=variance_epsilon)

            outputs.set_shape(inputs_shape)
            let outputs = match activation_fn with | None -> outputs | Some(f) -> f.Activate(outputs)
            utils.collect_named_outputs(defaultArg output_collections Array.empty<string>, vs.name, outputs)
            )


type Utils() =
    /// <summary>Perform dropout.</summary>
    /// <param name="input_tensor">float Tensor.</param>
    /// <param name="dropout_prob">float. The probability of dropping out a value 
    /// (NOT of *keeping* a value as in `tf.nn.dropout`).</param>
    /// <returns>A version of `input_tensor` with dropout applied</returns>
    static member dropout(input_tensor, dropout_prob : float32) =
        if dropout_prob = 0.0f
        then input_tensor
        else tf.nn.dropout(input_tensor, tf.constant(1.0f - dropout_prob))
    /// Run layer normalization on the last dimension of the tensor."""
    static member layer_norm(input_tensor) =
      Layers.layer_norm(inputs=input_tensor, begin_norm_axis = -1, begin_params_axis = -1)

    
    /// <summary>Runs layer normalization followed by dropout.</summary>
    static member layer_norm_and_dropout(input_tensor: Tensor, dropout_prob: float32) =
        Utils.dropout(Utils.layer_norm(input_tensor),dropout_prob)

    /// <summary>Creates a `truncated_normal_initializer` with the given range.</summary>
    static member create_initializer(?initializer_range: float32) =
        tf.truncated_normal_initializer(stddev = defaultArg initializer_range 0.02f)

/// <summary>Gaussian Error Linear Unit.
///  This is a smoother version of the RELU.
///  Original paper: https://arxiv.org/abs/1606.08415
/// </summary>
/// <param name="x">float Tensor to perform activation.</param>
/// <returns>  `x` with the GELU activation applied.</returns>
let gelu(x: Tensor) =
    let cdf = 0.5 * (1.0 + tf.tanh(0.7978845608 * (x + 0.044715 * tf.pow(x, 3))))
    x * cdf

module Activation = 
    let Gelu = {new Operations.Activation.IActivation with member this.Activate(x,_) = gelu(x)}
    let Relu  = Operations.Activation.relu()
    let Tanh  = Operations.Activation.tanh()
    let Linear = Operations.Activation.linear()


/// Configuration for `BertModel`.
type BertConfig = {
    /// Vocabulary size of `inputs_ids` in `BertModel`.
    vocab_size : int option
    /// Size of the encoder layers and the pooler layer.
    hidden_size : int
    /// Number of hidden layers in the Transformer encoder.
    num_hidden_layers : int
    /// Number of attention heads for each attention layer in
    /// the Transformer encoder.
    num_attention_heads : int
    /// The size of the "intermediate" (i.e., feed-forward)
    /// layer in the Transformer encoder.
    intermediate_size : int
    /// The non-linear activation function (function or string) in the
    /// encoder and pooler.
    hidden_act : IActivation
    /// The dropout probability for all fully connected
    /// layers in the embeddings, encoder, and pooler.
    hidden_dropout_prob : float32
    ///  The dropout ratio for the attention
    ///  probabilities.
    attention_probs_dropout_prob : float32
    ///  The maximum sequence length that this model might
    ///    ever be used with. Typically set this to something large just in case
    ///    (e.g., 512 or 1024 or 2048).
    max_position_embeddings : int
    /// The vocabulary size of the `token_type_ids` passed into
    /// `BertModel`.
    type_vocab_size : int
    // The stdev of the truncated_normal_initializer for
    initializer_range : float32
} with 
    static member Default = 
        {
            vocab_size = None
            hidden_size = 768
            num_hidden_layers = 12
            num_attention_heads = 12
            intermediate_size = 3072
            hidden_act = Activation.Gelu
            hidden_dropout_prob = 0.1f
            attention_probs_dropout_prob = 0.1f
            max_position_embeddings = 512
            type_vocab_size = 16
            initializer_range = 0.02f
         }

    /// Constructs a `BertConfig` from a Python dictionary of parameters.
    static member from_dict(json_object : string) = failwith "todo"
    /// Constructs a `BertConfig` from a json file of parameters.
    static member from_file(json_file : string) = failwith "todo"
    /// Serializes this instance to a Python dictionary.
    static member to_dict() = failwith "todo" 
    /// Serializes this instance to a JSON string.
    static member to_json_string() = failwith "todo"

/// <summary>
/// BERT model ("Bidirectional Encoder Representations from Transformers").
///  Example usage:
///  ```python
///  # Already been converted into WordPiece token ids
///  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
///  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
///  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
///  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
///    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
///  model = modeling.BertModel(config=config, is_training=True,
///    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
///  label_embeddings = tf.get_variable(...)
///  pooled_output = model.get_pooled_output()
///  logits = tf.matmul(pooled_output, label_embeddings)
///  ...
/// </summary>
/// <param name="config"> `BertConfig` instance.</param>
/// <param name="is_training"> bool. true for training model, false for eval model. Controls 
/// whether dropout will be applied.</param>
/// <param name="input_ids">int32 Tensor of shape [batch_size, seq_length].</param>
/// <param name="input_mask"> (optional) int32 Tensor of shape [batch_size, seq_length].</param>
/// <param name="token_type_ids"> (optional) int32 Tensor of shape [batch_size, seq_length].</param>
/// <param name="use_one_hot_embeddings"> (optional) bool. Whether to use one-hot word
///        embeddings or tf.embedding_lookup() for the word embeddings.</param>
/// <param name="scope"> (optional) variable scope. Defaults to "bert".</param>
/// <exception cref="Tensorflow.ValueError"> The config is invalid or one of the input tensor shapes
/// is invalid </exception>
type BertModel(config: BertConfig, 
               is_training: bool, 
               input_ids: Tensor,
               ?input_mask: Tensor,
               ?token_type_ids: Tensor,
               ?use_one_hot_embeddings: bool,
               ?scope: string) = 
    let scope = defaultArg scope "bert"
    let use_one_hot_embeddings = defaultArg use_one_hot_embeddings false
    let config = 
        if not is_training 
        then {config with hidden_dropout_prob = 0.0f; attention_probs_dropout_prob = 0.0f}
        else config

    let input_shape : int[] = BertModel.get_shape_list(input_ids, expected_rank=2)
    let batch_size = input_shape.[0]
    let seq_length = input_shape.[1]

    let input_mask = defaultArg input_mask (tf.ones(TensorShape(batch_size, seq_length),dtype=tf.int32))
    let token_type_ids = defaultArg token_type_ids (tf.zeros(TensorShape(batch_size, seq_length),dtype=tf.int32))
    let vocab_size = defaultArg config.vocab_size -1 // TODO figure out if this should be an error

    let (pooled_output, sequence_output, all_encoder_layers, embedding_table, embedding_output) =
        Tensorflow.Binding.tf_with(
            tf.variable_scope(scope, default_name="bert"),
            fun _ ->
            let (embedding_output, embedding_table) = 
                Tensorflow.Binding.tf_with(tf.variable_scope("embeddings"), fun _ ->
                    let (embedding_output, embedding_table) =
                        BertModel.embedding_lookup(
                            input_ids=input_ids,
                            vocab_size=vocab_size,
                            embedding_size=config.hidden_size,
                            initializer_range=config.initializer_range,
                            word_embedding_name="word_embeddings",
                            use_one_hot_embeddings=use_one_hot_embeddings)

                    let embedding_output = 
                        BertModel.embedding_postprocessor(
                            input_tensor=embedding_output,
                            use_token_type=true,
                            token_type_ids=token_type_ids,
                            token_type_vocab_size=config.type_vocab_size,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=true,
                            position_embedding_name="position_embeddings",
                            initializer_range=config.initializer_range,
                            max_position_embeddings=config.max_position_embeddings,
                            dropout_prob=config.hidden_dropout_prob)

                    (embedding_output, embedding_table))

            //This converts a 2D mask of shape [batch_size, seq_length] to a 3D
            // mask of shape [batch_size, seq_length, seq_length] which is used
            // for the attention scores.
            let all_encoder_layers =
                Tensorflow.Binding.tf_with(tf.variable_scope("encoder"), fun _ ->
                    let attention_mask = BertModel.create_attention_mask_from_input_mask(input_ids, input_mask)
                    // Run the stacked transformer.
                    // `sequence_output` shape = [batch_size, seq_length, hidden_size].
                    BertModel.transformer_model(input_tensor=embedding_output,
                                                attention_mask=attention_mask,
                                                hidden_size=config.hidden_size,
                                                num_hidden_layers=config.num_hidden_layers,
                                                num_attention_heads=config.num_attention_heads,
                                                intermediate_size=config.intermediate_size,
                                                intermediate_act_fn=config.hidden_act,
                                                hidden_dropout_prob=config.hidden_dropout_prob,
                                                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                                initializer_range=config.initializer_range,
                                                do_return_all_layers=true))

            let sequence_output = all_encoder_layers |> Seq.last
            // The "pooler" converts the encoded sequence tensor of shape
            // [batch_size, seq_length, hidden_size] to a tensor of shape
            // [batch_size, hidden_size]. This is necessary for segment-level
            // (or segment-pair-level) classification tasks where we need a fixed
            // dimensional representation of the segment.
            // We "pool" the model by simply taking the hidden state corresponding
            // to the first token. We assume that this has been pre-trained
            let pooled_output = 
                Tensorflow.Binding.tf_with(tf.variable_scope("pooler"), fun _ -> 
                    let first_token_tensor = tf.squeeze(sequence_output.[Slice.All, Slice(Nullable(0),Nullable(1)), Slice.All], axis=[|1|])
                    Layers.dense(first_token_tensor,
                                    config.hidden_size,
                                    activation=tanh(),
                                    kernel_initializer=Utils.create_initializer(config.initializer_range)))
            (pooled_output, sequence_output, all_encoder_layers, embedding_table, embedding_output))

    

    with 
        member this.PooledOutput = pooled_output

        /// <summary>Gets final hidden layer of encoder</summary>
        /// <returns>float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
        /// to the final hidden of the transformer encoder.</returns>
        member this.SequenceOutput = sequence_output

        member this.AllEncoderLayers = all_encoder_layers

        /// <summary>Gets output of the embedding lookup (i.e., input to the transformer).</summary>
        /// <returns> float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
        /// to the output of the embedding layer, after summing the word
        /// embeddings with the positional embeddings and the token type embeddings,
        /// then performing layer normalization. This is the input to the transformer.  </returns>
        member this.EmbeddingOutput = embedding_output

        member this.EmbeddingTable = embedding_table

        /// Compute the union of the current variables and checkpoint variables.
        static member get_assignment_map_from_checkpoint(tvars:RefVariable[], init_checkpoint) =

            let re = System.Text.RegularExpressions.Regex("^(.*):\\d+$")
            
            let name_to_variable = 
                [|
                    for var in tvars do
                        let name = var.name
                        let m = re.Match(name)
                        if m.Groups.Count > 1 then 
                            yield (m.Groups.[1].Value,var)
                |] 
                //|> Map.ofArray // NOTE the Map to RefVariable is not used     
                |> Array.map fst |> Set


            //let init_vars = tf.train.list_variables(init_checkpoint)
            //let xs = init_vars |> Array.map fst |> Array.filter name_to_variable.ContainsKey

            // NOTE: Expects full path to *.ckpt.meta file, this may not be correct
            let list_variables(path : string) =
                use f = System.IO.File.OpenRead(path)
                let metaGraph = Tensorflow.MetaGraphDef.Parser.ParseFrom(f)
                metaGraph.GraphDef.Node 
                |> Seq.choose (fun x -> if x.Op = "VariableV2" then Some(x.Name) else None)
                |> Seq.toArray

            let variable_names : string[] = list_variables(init_checkpoint)
            let xs = variable_names |> Array.filter name_to_variable.Contains
            let assignment_map = Map([| for x in xs -> (x,x)|])
            let initialized_variable_names = Map([| for x in xs do yield (x,1); yield (x + ":0",1)|])
            
            (assignment_map, initialized_variable_names)

         /// <summary>Create 3D attention mask from a 2D tensor mask.</summary>
         /// <param name="from_tensor"> 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].</param>
         /// <param name="to_mask"> int32 Tensor of shape [batch_size, to_seq_length].</param>
         /// <returns>float Tensor of shape [batch_size, from_seq_length, to_seq_length].</returns>
        static member create_attention_mask_from_input_mask(from_tensor, to_mask) =
            let from_shape = BertModel.get_shape_list(from_tensor, expected_rank=[|2; 3|])
            let batch_size = from_shape.[0]
            let from_seq_length = from_shape.[1]
            
            let to_shape = BertModel.get_shape_list(to_mask, expected_rank=2)
            let to_seq_length = to_shape.[1]
            
            let to_mask = 
                tf.cast(tf.reshape(to_mask, [|batch_size; 1; to_seq_length|]), tf.float32)
            
            //  We don't assume that `from_tensor` is a mask (although it could be). We
            //  don't actually care if we attend *from* padding tokens (only *to* padding)
            //  tokens so we create a tensor of all ones.
            // 
            // `broadcast_ones` = [batch_size, from_seq_length, 1]
            let broadcast_ones = tf.ones(shape=TensorShape(batch_size, from_seq_length, 1), dtype=tf.float32)
            //
            //   Here we broadcast along two dimensions to create the mask.
            let mask = broadcast_ones * to_mask
            mask

        /// <summary>Multi-headed, multi-layer Transformer from "Attention is All You Need".
        ///  This is almost an exact implementation of the original Transformer encoder.
        ///  See the original paper:
        ///  https://arxiv.org/abs/1706.03762
        ///  Also see:
        ///  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
        /// </summary>
        /// <param name="input_tensor"> float Tensor of shape [batch_size, seq_length, hidden_size].</param>
        /// <param name="attention_mask"> (optional) int32 Tensor of shape [batch_size, seq_length,
        ///      seq_length], with 1 for positions that can be attended to and 0 in
        ///      positions that should not be.</param>
        /// <param name="hidden_size"> int. Hidden size of the Transformer.</param>
        /// <param name="num_hidden_layers"> int. Number of layers (blocks) in the Transformer.</param>
        /// <param name="num_attention_heads"> int. Number of attention heads in the Transformer.</param>
        /// <param name="intermediate_size"> int. The size of the "intermediate" (a.k.a., feed
        ///      forward) layer.</param>
        /// <param name="intermediate_act_fn"> function. The non-linear activation function to apply
        ///      to the output of the intermediate/feed-forward layer.</param>
        /// <param name="hidden_dropout_prob"> float. Dropout probability for the hidden layers.
        ///    attention_probs_dropout_prob: float. Dropout probability of the attention
        ///      probabilities.</param>
        /// <param name="initializer_range"> float. Range of the initializer (stddev of truncated
        ///      normal).</param>
        /// <param name="do_return_all_layers"> Whether to also return all layers or just the final
        ///      layer.</param>
        ///  <returns>float Tensor of shape [batch_size, seq_length, hidden_size], the final
        ///    hidden layer of the Transformer.</returns>
        /// <exception cdef="Tensorflow.ValueError">A Tensor shape or paramter is invalid</exception>
        static member transformer_model(input_tensor : Tensor,
                                        ?attention_mask,
                                        ?hidden_size : int,
                                        ?num_hidden_layers : int,
                                        ?num_attention_heads : int,
                                        ?intermediate_size : int,
                                        ?intermediate_act_fn : Operations.Activation.IActivation,
                                        ?hidden_dropout_prob : float32,
                                        ?attention_probs_dropout_prob : float32,
                                        ?initializer_range : float32,
                                        ?do_return_all_layers: bool
                                       ) : Tensor[] = 
            let hidden_size = defaultArg hidden_size 768
            let num_hidden_layers = defaultArg num_hidden_layers 12
            let num_attention_heads = defaultArg num_attention_heads 12
            let intermediate_size = defaultArg intermediate_size 3072
            let intermediate_act_fn = defaultArg intermediate_act_fn (Activation.Gelu)
            let hidden_dropout_prob = defaultArg hidden_dropout_prob 0.1f
            let attention_probs_dropout_prob = defaultArg attention_probs_dropout_prob 0.1f
            let initializer_range = defaultArg initializer_range 0.02f
            let do_return_all_layers = defaultArg do_return_all_layers false
            if hidden_size % num_attention_heads <> 0 then
                raise (ValueError( sprintf "The hidden size (%d) is not a multiple of the number of attention heads (%d)" hidden_size num_attention_heads )) 

            let attention_head_size = hidden_size / num_attention_heads

            let input_shape = BertModel.get_shape_list(input_tensor, expected_rank=3)
            let batch_size = input_shape.[0]
            let seq_length = input_shape.[1]
            let input_width = input_shape.[2]

            /// The Transformer performs sum residuals on all layers so the input needs
            ///  to be the same as the hidden size.
            if input_width <> hidden_size then
                raise (ValueError(sprintf "The width of the input tensor (%d) != hidden size (%d)" input_width hidden_size))

            // We keep the representation as a 2D tensor to avoid re-shaping it back and
            // forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
            // the GPU/CPU but may not be free on the TPU, so we want to minimize them to
            // help the optimizer.
            let mutable prev_output = BertModel.reshape_to_matrix(input_tensor)

            let makeLayer(layer_idx : int) = 
                Tensorflow.Binding.tf_with(tf.variable_scope(sprintf "layer_%d" layer_idx), fun _ -> 
                    let layer_input = prev_output
                    let attention_output = 
                        Tensorflow.Binding.tf_with(tf.variable_scope("attention"), fun _ ->
                            // TODO/WARN The python code does not seem to be able to return
                            // more than one attention head here so either I'm wrong about that
                            // or a fair amount of the original code here is moot
                            let attention_output = 
                                Tensorflow.Binding.tf_with(tf.variable_scope("self"), fun _ ->
                                    BertModel.attention_layer(from_tensor=layer_input,
                                                              to_tensor=layer_input,
                                                              ?attention_mask=attention_mask,
                                                              num_attention_heads=num_attention_heads,
                                                              size_per_head=attention_head_size,
                                                              attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                              initializer_range=initializer_range,
                                                              do_return_2d_tensor=true,
                                                              batch_size=batch_size,
                                                              from_seq_length=seq_length,
                                                              to_seq_length=seq_length))
                            // Run a linear projection of `hidden_size` then add a residual
                            // with `layer_input`. 
                            let attention_output = 
                                Tensorflow.Binding.tf_with(tf.variable_scope("output"), fun _ ->
                                    let attention_output = Layers.dense(attention_output, 
                                                                           hidden_size,
                                                                           kernel_initializer=Utils.create_initializer(initializer_range))
                                    let attention_output = Utils.dropout(attention_output, hidden_dropout_prob)
                                    let attention_output = Utils.layer_norm(attention_output + layer_input)
                                    attention_output)
                            attention_output)

                    let intermediate_output = 
                            Tensorflow.Binding.tf_with(tf.variable_scope("intermediate"), fun _ ->
                                Layers.dense(attention_output, 
                                                intermediate_size,
                                                activation=intermediate_act_fn,
                                                kernel_initializer=Utils.create_initializer(initializer_range)))

                    let layer_output = 
                        // Down-project back to `hidden_size` then add the residual.
                        Tensorflow.Binding.tf_with(tf.variable_scope("output"), fun _ -> 
                            let layer_output = Layers.dense(intermediate_output, 
                                                               hidden_size,
                                                               kernel_initializer=Utils.create_initializer(initializer_range))
                            Utils.dropout(layer_output, hidden_dropout_prob)
                        )
                    prev_output <- layer_output
                    layer_output
                )

            let all_layer_outputs = [| for layer_idx in 0..num_hidden_layers - 1 -> makeLayer(layer_idx) |]

            if do_return_all_layers then
                [| for x in all_layer_outputs -> BertModel.reshape_from_matrix(x, input_shape)|]
            else
                [|BertModel.reshape_from_matrix(prev_output, input_shape)|]

        /// Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix).
        static member reshape_to_matrix(input_tensor) = 
            let ndims = input_tensor.TensorShape.ndim
            if ndims < 2 then 
                raise (ValueError(sprintf "Input tensor must have a least rank 2. Shape = %A" input_tensor.shape))
            elif ndims = 2 then input_tensor
            else 
                let width = input_tensor.shape |> Seq.last
                let output_tensor = tf.reshape(input_tensor, [|-1;width|])
                output_tensor

        /// Reshapes a rank 2 tensor back to its original rank >= 2 tensor.
        static member reshape_from_matrix(output_tensor, orig_shape_list) = 
            if orig_shape_list.Length = 2 
            then output_tensor
            else
                let output_shape = BertModel.get_shape_list(output_tensor)
                let orig_dims = orig_shape_list.[0..orig_shape_list.Length - 2]
                let width = output_shape |> Seq.last
                tf.reshape(output_tensor, Array.append orig_dims [|width|])

        ///<summary>Looks up word embeddings for id tensor.</summary>
        ///<param name="input_ids">int32 Tensor of shape [batch_size, seq_length] containing word ids.</param>
        ///<param name="vocab_size">int. Size of the embedding vocabulary.</param>
        ///<param name="embedding_size">int. Width of the word embeddings.</param>
        ///<param name="initializer_range">float. Embedding initialization range.</param>
        ///<param name="word_embedding_name">string. Name of the embedding table.</param>
        ///<param name="use_one_hot_embeddings">bool. If True, use one-hot method for word 
        /// embeddings. If False, use `tf.gather()`.</param>
        /// <returns> float Tensor of shape [batch_size, seq_length, embedding_size]. </returns>
        static member embedding_lookup(input_ids: Tensor, 
                                       vocab_size: int,
                                       ?embedding_size: int,
                                       ?initializer_range: float32,
                                       ?word_embedding_name: string,
                                       ?use_one_hot_embeddings: bool) = 
            let embedding_size = defaultArg embedding_size 128
            let initializer_range = defaultArg initializer_range 0.02f
            let word_embedding_name = defaultArg word_embedding_name "word_embeddings"
            let use_one_hot_embeddings = defaultArg use_one_hot_embeddings false
            // This function assumes that the input is of shape [batch_size, seq_length,
            // num_inputs].
            //
            // If the input is a 2D tensor of shape [batch_size, seq_length], we
            // reshape to [batch_size, seq_length, 1]. 
            let input_ids = 
                if input_ids.TensorShape.ndim = 2 
                then tf.expand_dims(input_ids, axis = -1) 
                else input_ids

            let embedding_table = 
                tf.get_variable(
                                name=word_embedding_name, 
                                shape=TensorShape(vocab_size, embedding_size),
                                initializer=Utils.create_initializer(initializer_range)
                                )._AsTensor()

            let flat_input_ids = tf.reshape(input_ids, [|-1|])
            let output = 
                if use_one_hot_embeddings then
                    tf.matmul2(tf.one_hot(flat_input_ids, depth=vocab_size), embedding_table)
                else
                    tf.gather(embedding_table, flat_input_ids)

            let input_shape = BertModel.get_shape_list(input_ids)

            let output = tf.reshape(output, [|yield! input_shape.[0..input_shape.Length-2]; yield input_shape.[input_shape.Length-1] * embedding_size|])
            (output, embedding_table)


        /// <summary>Performs various post-processing on a word embedding tensor.</summary>
        /// <param name="input_tensor">float Tensor of shape [batch_size, seq_length, 
        /// embedding_size].</param>
        /// <param name="use_token_type"> bool. Whether to add embeddings for 
        /// `token_type_ids`.</param>
        /// <param name="token_type_ids">(optional) int32 Tensor of shape [batch_size, seq_length].
        /// Must be specified if `use_token_type` is True. </param>
        /// <param name="token_type_vocab_size"> int. The vocabulary size of `token_type_ids`.</param>
        /// <param name="token_type_embedding_name"> string. The name of the embedding table variable
        /// for token type ids.</param>
        /// <param name="use_position_embeddings"> bool. Whether to add position embeddings for the
        /// position of each token in the sequence.</param>
        /// <param name="position_embedding_name"> string. The name of the embedding table variable
        /// for positional embeddings.</param>
        /// <param name="initializer_range"> float. Range of the weight initialization.</param>
        /// <param name="max_position_embeddings"> int. Maximum sequence length that might ever be
        /// used with this model. This can be longer than the sequence length of
        /// input_tensor, but cannot be shorter.</param>
        /// <param name="dropout_prob"> float. Dropout probability applied to the final output tensor.</param>
        /// <returns>float tensor with same shape as `input_tensor`</returns>
        /// <exception cdef="Tensorflow.ValueError">One of the tensor shapes or input values is invalid.</exception>
        static member embedding_postprocessor(input_tensor: Tensor,
                                          ?use_token_type: bool,
                                          ?token_type_ids: Tensor,
                                          ?token_type_vocab_size: int,
                                          ?token_type_embedding_name: string,
                                          ?use_position_embeddings: bool,
                                          ?position_embedding_name: string,
                                          ?initializer_range: float32,
                                          ?max_position_embeddings: int,
                                          ?dropout_prob: float32) = 
            
            let use_token_type = defaultArg use_token_type false
            //let token_type_ids = defaultArg token_type_ids None
            let token_type_vocab_size = defaultArg token_type_vocab_size 16
            let token_type_embedding_name = defaultArg token_type_embedding_name "token_type_embeddings"
            let use_position_embeddings = defaultArg use_position_embeddings  true
            let position_embedding_name = defaultArg position_embedding_name  "position_embeddings"
            let initializer_range = defaultArg initializer_range 0.02f
            let max_position_embeddings = defaultArg max_position_embeddings 512
            let dropout_prob = defaultArg dropout_prob 0.1f
            let input_shape = BertModel.get_shape_list(input_tensor, expected_rank=3)
            let batch_size = input_shape.[0] 
            let seq_length = input_shape.[1] 
            let width = input_shape.[2] 

            let output = input_tensor

            let output = 
                if use_token_type then
                    match token_type_ids with
                    | None -> raise (ValueError("`token_type_ids` must be specified if `use_token_type` is true."))
                    | Some(token_type_ids) ->
                        let token_type_table = 
                            let ttt = tf.get_variable(token_type_embedding_name,
                                                        dtype = tf.float32,
                                                        shape = TensorShape(token_type_vocab_size, width),
                                                        initializer = Utils.create_initializer(initializer_range)
                                                        )._AsTensor()
                            ttt
                            
                        
                        // This vocab will be small so we always do one-hot here, since it is always
                        // faster for a small vocabulary.
                        let flat_token_type_ids = tf.reshape(token_type_ids, [|-1|])
                        let one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
                        let token_type_embeddings = tf.matmul2(one_hot_ids, token_type_table)
                        let token_type_embeddings = 
                            tf.reshape(token_type_embeddings, [|batch_size; seq_length; width|])
                        output + token_type_embeddings
                else output

            let output = 
                if use_position_embeddings then
                    // TODO add an assert_less_equal
                    //let assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
                    let assert_op = tf.assert_equal(seq_length, max_position_embeddings)
                    use _cd = tf.control_dependencies([|assert_op|])
                    let full_position_embeddings = 
                        tf.get_variable(name = position_embedding_name,
                                        shape = TensorShape(max_position_embeddings, width),
                                        initializer = Utils.create_initializer(initializer_range)
                                        )._AsTensor()
                    // Since the position embedding table is a learned variable, we create it
                    // using a (long) sequence length `max_position_embeddings`. The actual
                    // sequence length might be shorter than this, for faster training of
                    // tasks that do not have long sequences.
                    //
                    // So `full_position_embeddings` is effectively an embedding table
                    // for position [0, 1, 2, ..., max_position_embeddings-1], and the current
                    // sequence has positions [0, 1, 2, ... seq_length-1], so we can just
                    // perform a slice.
                    let position_embeddings = tf.slice(full_position_embeddings, [|0;0|],[|seq_length;-1|])

                    let num_dims = output.TensorShape.as_list().Length

                    // Only the last two dimensions are relevant (`seq_length` and `width`), so
                    // we broadcast among the first dimensions, which is typically just
                    // the batch size.
                    let position_broadcast_shape = 
                         [| 
                            for _i in 0..num_dims - 3 do // todo double check "- 2"
                                yield 1
                            yield seq_length
                            yield width
                         |]
                    let position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
                    output + position_embeddings
                else output
            Utils.layer_norm_and_dropout(output, dropout_prob)

        /// <summary>Performs multi-headed attention from `from_tensor` to `to_tensor`.
        ///  This is an implementation of multi-headed attention based on "Attention
        ///  is all you Need". If `from_tensor` and `to_tensor` are the same, then
        ///  this is self-attention. Each timestep in `from_tensor` attends to the
        ///  corresponding sequence in `to_tensor`, and returns a fixed-with vector.
        ///  This function first projects `from_tensor` into a "query" tensor and
        ///  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
        ///  of tensors of length `num_attention_heads`, where each tensor is of shape
        ///  [batch_size, seq_length, size_per_head].
        ///  Then, the query and key tensors are dot-producted and scaled. These are
        ///  softmaxed to obtain attention probabilities. The value tensors are then
        ///  interpolated by these probabilities, then concatenated back to a single
        ///  tensor and returned.
        ///  In practice, the multi-headed attention are done with transposes and
        ///  reshapes rather than actual separate tensors.</summary>
        ///  <param name="from_tensor"> float Tensor of shape [batch_size, from_seq_length,
        ///      from_width].</param>
        ///  <param name="to_tensor"> float Tensor of shape [batch_size, to_seq_length, to_width].</param>
        ///  <param name="attention_mask"> (optional) int32 Tensor of shape [batch_size,
        ///      from_seq_length, to_seq_length]. The values should be 1 or 0. The
        ///      attention scores will effectively be set to -infinity for any positions in
        ///      the mask that are 0, and will be unchanged for positions that are 1.</param>
        ///  <param name="num_attention_heads"> int. Number of attention heads.</param>
        ///  <param name="size_per_head"> int. Size of each attention head.</param>
        ///  <param name="query_act"> (optional) Activation function for the query transform.</param>
        ///  <param name="key_act"> (optional) Activation function for the key transform.</param>
        ///  <param name="value_act"> (optional) Activation function for the value transform.</param>
        ///  <param name="attention_probs_dropout_prob"> (optional) float. Dropout probability of the
        ///      attention probabilities. </param>
        ///  <param name="initializer_range"> float. Range of the weight initializer.</param>
        ///  <param name="do_return_2d_tensor"> bool. If True, the output will be of shape [batch_size
        ///      * from_seq_length, num_attention_heads * size_per_head]. If False, the
        ///      output will be of shape [batch_size, from_seq_length, num_attention_heads
        ///      * size_per_head].
        ///    batch_size: (Optional) int. If the input is 2D, this might be the batch size
        ///      of the 3D version of the `from_tensor` and `to_tensor`.
        ///    from_seq_length: (Optional) If the input is 2D, this might be the seq length
        ///      of the 3D version of the `from_tensor`.
        ///    to_seq_length: (Optional) If the input is 2D, this might be the seq length
        ///      of the 3D version of the `to_tensor`. </param>
        ///  <returns>
        ///    float Tensor of shape [batch_size, from_seq_length,
        ///      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        ///      true, this will be of shape [batch_size * from_seq_length,
        ///      num_attention_heads * size_per_head]). </returns>
        /// <exception cdef="Tensorflow.ValueError">Any of the arguments or tensor shapes are invalid.</exception>
        static member attention_layer(
                                      from_tensor : Tensor, 
                                      to_tensor : Tensor, 
                                      ?attention_mask : Tensor,
                                      ?num_attention_heads : int,
                                      ?size_per_head : int,
                                      ?query_act : Operations.Activation.IActivation,
                                      ?key_act : Operations.Activation.IActivation,
                                      ?value_act : Operations.Activation.IActivation,
                                      ?attention_probs_dropout_prob : float32,
                                      ?initializer_range : float32,
                                      ?do_return_2d_tensor : bool,
                                      ?batch_size : int,
                                      ?from_seq_length : int,
                                      ?to_seq_length : int) = 
            
            let num_attention_heads = defaultArg num_attention_heads 1
            let size_per_head = defaultArg size_per_head 512
            let attention_probs_dropout_prob = defaultArg attention_probs_dropout_prob 0.0f
            let initializer_range = defaultArg initializer_range 0.02f
            let do_return_2d_tensor = defaultArg do_return_2d_tensor false
            let transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width) =
                let output_tensor = tf.reshape(input_tensor, [|batch_size; seq_length; num_attention_heads; width|])
                let output_tensor = tf.transpose(output_tensor, [|0; 2; 1; 3|])
                output_tensor

            let from_shape = BertModel.get_shape_list(from_tensor, expected_rank=[|2;3|])
            let to_shape = BertModel.get_shape_list(to_tensor, expected_rank=[|2;3|])
            if from_shape.Length <> to_shape.Length then
                raise (ValueError("The rank of `from_tensor` must match the rank of `to_tensor`."))

            let (batch_size, from_seq_length, to_seq_length) =
                match from_shape.Length with
                | 3 -> from_shape.[0], from_shape.[1], to_shape.[1]
                | 2 -> 
                    match batch_size, from_seq_length, to_seq_length with
                    | Some(x), Some(y), Some(z) -> (x,y,z)
                    | _,_,_ -> 
                        raise (ValueError("When passing in a rank 2 tensors to attention_layer, the values " +
                                          "for `batch_size`, `from_seq_length`, and `to_seq_length` " +
                                          "must all be specified."))
                | n -> failwithf "from_shape is expected to be 3 or 2 but is %i" n

            // Scalar dimensions referenced here:
            //   B = batch size (number of sequences)
            //   F = `from_tensor` sequence length
            //   T = `to_tensor` sequence length
            //   N = `num_attention_heads`
            //   H = `size_per_head`

            // TODO: Need to fix the ability to easily pass in an optional to a C# function
            //       e.g. tf.layers.dense(...,?activation=query)
            //       Then we can remove the identity activations from this code

            let identity = {
                            new Operations.Activation.IActivation 
                            with member this.Activate(x:Tensor,name:string) = tf.identity(x,name)
                            }

            let query_act = defaultArg query_act identity
            let key_act = defaultArg key_act identity
            let value_act = defaultArg value_act identity

            let from_tensor_2d = BertModel.reshape_to_matrix(from_tensor)
            let to_tensor_2d = BertModel.reshape_to_matrix(to_tensor)

            // `query_layer` = [B*F, N*H]
            let query_layer = Layers.dense(from_tensor_2d,
                                              num_attention_heads * size_per_head,
                                              activation=query_act,
                                              name="query",
                                              kernel_initializer=Utils.create_initializer(initializer_range))

            // `key_layer` = [B*T, N*H]
            let key_layer = Layers.dense(to_tensor_2d,
                                            num_attention_heads * size_per_head,
                                            activation=key_act,
                                            name="key",
                                            kernel_initializer=Utils.create_initializer(initializer_range))

            // `value_layer` = [B*T, N*H]
            let value_layer = Layers.dense(to_tensor_2d,
                                              num_attention_heads * size_per_head,
                                              activation=value_act,
                                              name="value",
                                              kernel_initializer=Utils.create_initializer(initializer_range))

            // `query_layer` = [B, N, F, H]
            let query_layer = transpose_for_scores(query_layer, batch_size, 
                                                   num_attention_heads, from_seq_length,
                                                   size_per_head)

            // `key_layer` = [B, N, T, H]
            let key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                                 to_seq_length, size_per_head)

            // Take the dot product between "query" and "key" to get the raw
            // attention scores.
            // `attention_scores` = [B, N, F, T]
            let attention_scores = tf.matmul2(query_layer, key_layer, transpose_b=true)
            let attention_scores = tf.multiply(attention_scores, 1.0 / Math.Sqrt(float size_per_head))

            let attention_scores = 
                match attention_mask with
                | None -> attention_scores
                | Some(attention_mask) ->
                    // `attention mask` = [B, 1, F, T]
                    // TODO We might be able to do something like attention_maks.[:,None]
                    let attention_mask = tf.expand_dims(attention_mask, axis=1)

                    // Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                    // masked positions, this operation will create a tensor which is 0.0 for
                    // positions we want to attend and -10000.0 for masked positions.
                    let adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

                    // Since we are adding it to the raw scores before the softmax, this is
                    // effectively the same as removing these entirely.
                    attention_scores + adder

            // Normalize the attention scores to probabilities.
            // `attention_probs` = [B, N, F, T]
            let attention_probs = tf.nn.softmax(attention_scores)

            // This is actually dropping out entire tokens to attend to, which might
            // seem a bit unusual, but is taken from the original Transformer paper
            let attention_probs = Utils.dropout(attention_probs, attention_probs_dropout_prob)

            // `value_layer` = [B, T, N, H]
            let value_layer = 
                tf.reshape(value_layer, 
                           [|batch_size; to_seq_length; num_attention_heads; size_per_head|])

            // `value_layer` = [B, N, T, H]
            let value_layer = tf.transpose(value_layer, [|0; 2; 1; 3|])

            // `context_layer` = [B, N, F, H]
            let context_layer = tf.matmul2(attention_probs, value_layer)

            // `context_layer` = [B, F, N, H]
            let context_layer = tf.transpose(context_layer, [|0; 2; 1; 3|])

            let context_layer = 
                if do_return_2d_tensor then
                    // `context_layer` = [B*F, N*H]
                    tf.reshape(context_layer,
                               [|batch_size * from_seq_length; num_attention_heads * size_per_head|])
                else
                    // `context_layer` = [B, F, N*H]
                    tf.reshape(context_layer,
                               [|batch_size; from_seq_length; num_attention_heads * size_per_head|])

            context_layer

        /// <summary>Raises an exception if the tensor rank is not of the expected rank.</summary>
        /// <param name="tensor">A tf.Tensor to check the rank of.</param>
        /// <param name="expected_rank">list of integers, expected rank.</param>
        /// <param name="name">Optional name of the tensor for the error message.<param>
        /// <exception cdef="Tensorflow.ValueError">If the expected shape doesn't match the actual shape.</exception>
        static member assert_rank(tensor: Tensor, expected_rank: int[], ?name: string) =
            let name = defaultArg name tensor.name
            let expected_rank_dict = set expected_rank 
            let actual_rank = tensor.TensorShape.ndim
            if not(expected_rank.Contains(actual_rank)) then
                let scope_name = tf.get_variable_scope().name
                raise (ValueError(sprintf "For the tensor.`%s` in scope `%s`, the actual rank `%d` (shape = %O) is not equal to the expected rank `%A`"
                    name scope_name actual_rank tensor.TensorShape expected_rank))

        /// <summary>Raises an exception if the tensor rank is not of the expected rank.</summary>
        /// <param name="tensor">A tf.Tensor to check the rank of.</param>
        /// <param name="expected_rank">int, expected rank.</param>
        /// <param name="name">Optional name of the tensor for the error message.<param>
        /// <exception cdef="Tensorflow.ValueError">If the expected shape doesn't match the actual shape.</exception>
        static member assert_rank(tensor: Tensor, expected_rank: int, ?name: string) =
            BertModel.assert_rank(tensor, [|expected_rank|],?name=name)

        /// <summary>Returns a list of the shape of tensor, preferring static dimensions.</summary>
        /// <param name="tensor">A tf.Tensor object to find the shape of.</param>
        /// <param name="expected_rank"> (optional) int. The expected rank of `tensor`. If this is
        ///   specified and the `tensor` has a different rank, and exception will be
        ///   thrown.</param>
        /// <param name="name"> Optional name of the tensor for the error message.</param>
        /// <returns>
        /// A list of dimensions of the shape of tensor. All static dimensions will
        /// be returned as python integers, and dynamic dimensions will be returned
        /// as tf.Tensor scalars.
        /// </returns>
        static member get_shape_list(tensor: Tensor, ?expected_rank: int, ?name: string) : int[] =
            BertModel.get_shape_list(tensor, expected_rank |> Option.toArray, ?name = name )

//        static member get_shape_list(tensor: Tensor) =
//            BertModel.get_shape_list(tensor,[||])

        /// <summary>Returns a list of the shape of tensor, preferring static dimensions.</summary>
        /// <param name="tensor">A tf.Tensor object to find the shape of.</param>
        /// <param name="expected_rank"> (optional) int. The expected rank of `tensor`. If this is
        ///   specified and the `tensor` has a different rank, and exception will be
        ///   thrown.</param>
        /// <param name="name"> Optional name of the tensor for the error message.</param>
        /// <returns>
        /// A list of dimensions of the shape of tensor. All static dimensions will
        /// be returned as python integers, and dynamic dimensions will be returned
        /// as tf.Tensor scalars.
        /// </returns>
        static member get_shape_list(tensor: Tensor, expected_rank: int[], ?name: string) : int[] =
            let name = defaultArg name tensor.name
            //expected_rank |> Option.iter (fun expected_rank -> BertModel.assert_rank(tensor, expected_rank,name))
            if expected_rank.Length > 0 then 
                BertModel.assert_rank(tensor, expected_rank,name)

            let shape = tensor.TensorShape.as_list()
            if shape |> Array.exists (fun x -> x < 0)
            then
                //let dyn_shape = tf.shape(tensor)
                //shape |> Array.mapi (fun i x -> if x < 0 then Choice2Of2(dyn_shape.[i]) else Choice1Of2(i))
                failwith "Non-static shapes are not supported at this time"
            else 
                //shape |> Array.map Choice1Of2
                shape

