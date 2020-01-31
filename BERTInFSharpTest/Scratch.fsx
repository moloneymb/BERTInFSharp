// Testing tokenization

#I @"C:\Users\moloneymb\.nuget\packages\"
#r @"system.runtime.compilerservices.unsafe\4.5.2\lib\netstandard2.0\System.Runtime.CompilerServices.Unsafe.dll"
#r @"numsharp\0.20.5\lib\netstandard2.0\NumSharp.Core.dll"
#r @"tensorflow.net\0.14.0\lib\netstandard2.0\TensorFlow.NET.dll"
#r @"system.memory\4.5.3\lib\netstandard2.0\System.Memory.dll"
#r @"google.protobuf\3.10.1\lib\netstandard2.0\Google.Protobuf.dll"

#load @"..\BertInFSharp\utils.fs"
#load @"..\BertInFSharp\tokenization.fs"
//#load @"..\BertInFSharp\modeling.fs"

open Tokenization
open System
open System.IO

// TODO get shape


