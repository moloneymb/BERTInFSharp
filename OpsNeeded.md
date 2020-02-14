# These notes are for aiding a TF.Net translation to DiffSharp

## Tensor types used by BERT

* [ ] Float32
* [ ] Int32

No Quantization is used

## Base Operations needed for BERT Inference

This is a list of the base TensorFlow ops needed to run BERT. This list excludes higher level ops which are compositions of the base ops. The list also excludes trivial, or framework specific ops.


| Tensorflow 1.0    |    DiffSharp Node and/or [TensorOp](https://github.com/DiffSharp/DiffSharp/blob/dev/src/DiffSharp.Core/Tensor.fs#L973)    |  Notes           |
| ------------------|------------------|------------------|
| Add               |      `(+)`/AddTT       | Not all broadcasting is implemented         |
| BatchMatMul       |      MatMul      |  Batching is not supported                 |
| BiasAdd           |      `(+)`/AddTT         |  [TF biasAdd is same as add except on quantized types](https://www.tensorflow.org/api_docs/python/tf/nn/bias_add)     |
| Cast              |  **todo**          |                  |
| ExpandDims        |     Unsqueeze    |                  |
| GatherV2          |                  |                  |
| MatMul            |   MatMul         |  Batching not implemented      |
| Mean              |   Mean           |    keep_dims not available     |
| Merge             |                 |                  |
| Mul               |   `(*)`/MulTT    |  Not all broadcasting is implemented                |
| OneHot            |  **todo**                |                  |
| Pow               |  Pow             |  Not all broadcasting is implemented                 |
| Reshape           |  **todo**            |  Not yet available, can normally use `View` instead                 |
| Rsqrt             |  **todo** - add to API  |                  |
| Slice             |  Slice                |                  |
| Softmax           |  Softmax         |                  |
| SquaredDifference |  **todo** - add to API                |                  |
| Squeeze           |  Squeeze         |                  |
| StopGradient      |  n/a             | In DiffSharp these correspond to the different node types for Tensors |
| StridedSlice      |  Slice is ok     | Strides are 1 in BERT.  Note F# has no strided indexing syntax |
| Sub               |  `(-)`/SubTT     | broadcasting not implemented |
| Tanh              |  Tanh            |                  |
| Transpose         |  Transpose       | Transpose should accept dimensions to swap |

## Additional Base Operations needed for BERT training

| Tensorflow 1.0    |    DiffSharp     |  Notes           |
| ------------------|------------------|------------------|
| AddN              |                  |                  |
| ArgMax            |                  |                  |
| BatchMatMul       |                  |                  |
| BiasAdd           |                  |                  |
| BiasAddGrad       |                  |                  |
| BroadcastGradientArgs       |                  |                  |
| ConcatV2          |                  |                  |
| DynamicStitch     |                  |                  |
| FloorDiv          |                  |                  |
| FloorMod          |                  |                  |
| Greater           |                  |                  |
| GreaterEqual      |                  |                  |
| InvertPermutation       |                  |                  |
| IsFinite          |                  |                  |
| L2Loss            |                  |                  |
| Log               |                  |                  |
| Maximum           |                  |                  |
| Minimum           |                  |                  |
| Neg               |                  |                  |
| Pack              |                  |                  |
| Pad               |                  |                  |
| Prod              |                  |                  |
| Range             |                  |                  |
| RealDiv           |                  |                  |
| Reciprocal        |                  |                  |
| RsqrtGrad         |                  |                  |
| Sqrt              |                  |                  |
| Square            |                  |                  |
| StridedSlice      |                  |                  |
| StridedSliceGrad  |                  |                  |
| Sum               |                  |                  |
| Tanh              |                  |                  |
| TanhGrad          |                  |                  |
| Tile              |                  |                  |
| UnsortedSegmentSum    |                  |                  |

## Additional Base Operations needed for Asertions

| Tensorflow 1.0    |    DiffSharp     |  Notes           |
| ------------------|------------------|------------------|
| All               |                  |                  |
| Assert            |                  |                  |
| Equal             |                  |                  |
| Switch            |                  |                  |

