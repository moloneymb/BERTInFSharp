# These notes are for aiding a TF.Net translation to DiffSharp

## Base Operations needed for BERT Inference

This is a list of the base TensorFlow ops needed to run BERT. This list excludes higher level ops which are compositions of the base ops. The list also excludes trivial, or framework specific ops.

* Add
* BatchMatMul
* BiasAdd
* Cast
* ExpandDims 
* GatherV2
* MatMul
* Mean
* Merge
* Mul
* OneHot
* Pow
* Reshape
* Rsqrt
* Slice
* Softmax
* SquaredDifference
* Squeeze
* StopGradient
* StridedSlice 
* Sub
* Tanh
* Transpose

## Additional Base Operations needed for BERT training

* AddN
* ArgMax
* BatchMatMul
* BiasAdd
* BiasAddGrad
* BroadcastGradientArgs
* ConcatV2
* DynamicStitch
* FloorDiv
* FloorMod
* Greater
* GreaterEqual
* InvertPermutation
* IsFinite
* L2Loss
* Log
* Maximum
* Minimum
* Neg
* Pack
* Pad
* Prod
* Range
* RealDiv
* Reciprocal
* RsqrtGrad
* Sqrt
* Square
* StridedSlice
* StridedSliceGrad
* Sum
* Tanh
* TanhGrad
* Tile
* UnsortedSegmentSum

## Additional Base Operations needed for Asertions

* All
* Assert 
* Equal 
* Switch 