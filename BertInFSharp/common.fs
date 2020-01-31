module Common

open System.IO

let chkpt = @"C:\Users\moloneymb\Downloads\uncased_L-12_H-768_A-12\uncased_L-12_H-768_A-12"
let vocab_file = Path.Combine(chkpt, "vocab.txt")
let bert_config_file = Path.Combine(chkpt, "bert_config.json")
