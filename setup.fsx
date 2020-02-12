
#I @"/home/moloneymb/.nuget/packages/"


#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r @"numsharp/0.20.5/lib/netstandard2.0/NumSharp.Core.dll"
#r @"tensorflow.net/0.14.0/lib/netstandard2.0/TensorFlow.NET.dll"
#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
#r @"google.protobuf/3.10.1/lib/netstandard2.0/Google.Protobuf.dll"
#r @"argu/6.0.0/lib/netstandard2.0/Argu.dll"
#r @"csvhelper/12.2.3/lib/net47/CsvHelper.dll"
#r @"newtonsoft.json/12.0.2/lib/net45/Newtonsoft.Json.dll"
#r @"sharpziplib/1.2.0/lib/net45/ICSharpCode.SharpZipLib.dll"

#load @"BertInFSharp/common.fs"
Common.setup()
#load @"BertInFSharp/utils.fs"
#load @"BertInFSharp/tokenization.fs"
#load @"BertInFSharp/run_classifier.fs"
#load @"BertInFSharp/modeling.fs"
#load @"BertInFSharp/optimization.fs"

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



