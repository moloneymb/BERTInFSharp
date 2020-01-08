namespace BERTInFSharpTest

// TODO double check unicode handeling in .Net with regards to the python equivalent of u"word"

open System
open System.IO
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type TokenizationTestClass () =

    [<TestMethod>]
    member this.test_full_tokenizer() =
        let vocab_tokens = [| "[UNK]"; "[CLS]"; "[SEP]"; "want"; "##want"; "##ed"; "wa"; "un"; "runn"; "##ing"; "," |]
        let vocab_filename = IO.Path.GetTempFileName()
        File.WriteAllLines(vocab_filename, vocab_tokens)
        let tokenizer = Tokenization.FullTokenizer(vocab_filename)
        File.Delete(vocab_filename)
        let tokens = tokenizer.tokenize("UNwant\u00E9d,running")
        Assert.AreEqual(tokens, [|"un"; "##want"; "##ed"; ","; "runn"; "##ing"|]) |> ignore
        Assert.AreEqual(tokenizer.convert_tokens_to_ids(tokens), [|7; |])

    [<TestMethod>]
    member this.test_chinese() = 
        let tokenizer = Tokenization.BasicTokenizer()
        Assert.AreEqual(tokenizer.tokenize("ah\u535A\u63A8zz"),[|"ah"; "\u535A"; "\u63A8"; "zz"|])


    [<TestMethod>]
    member this.test_basic_tokenizer_lower() =
        let tokenizer = Tokenization.BasicTokenizer(do_lower_case = true)
        Assert.AreEqual(tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "),
            [|"hello"; "!"; "how"; "are"; "you"; "?"|])
        Assert.AreEqual(tokenizer.tokenize("H\u00E9llo"), [|"hello"|])

    [<TestMethod>]
    member this.test_basic_tokenizer_no_lower() = 
        let tokenizer = Tokenization.BasicTokenizer(do_lower_case = false)
        Assert.AreEqual(tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "),
            [|"HeLLo"; "!"; "how"; "Are"; "yoU"; "?"|])

    [<TestMethod>]
    member this.test_wordpiece_tokenizer() = 
        let vocab_tokens = [| "[UNK]"; "[CLS]"; "[SEP]"; "want"; "##want"; "##ed"; "wa"; "un"; "runn"; "##ing" |] 
        let vocab = vocab_tokens |> Array.mapi (fun i x -> (x,i)) |> dict
        let tokenizer = Tokenization.WordpieceTokenizer(vocab=vocab)
        Assert.AreEqual(tokenizer.tokenize(""), [||])
        Assert.AreEqual(tokenizer.tokenize("unwanted running"), [|"un"; "##want"; "##ed"; "runn"; "##ing"|])
        Assert.AreEqual(tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])
        
    [<TestMethod>]
    member this.test_convert_tokens_to_ids() = 
        let vocab_tokens = [| "[UNK]"; "[CLS]"; "[SEP]"; "want"; "##want"; "##ed"; "wa"; "un"; "runn"; "##ing" |] 
        let vocab = vocab_tokens |> Array.mapi (fun i x -> (x,i)) |> dict
        Assert.AreEqual(Tokenization.convert_tokens_to_ids(vocab, [|"un"; "##want"; "##ed"; "runn"; "##ing"|]), [|7; 4; 5; 8; 9|])

    [<TestMethod>]
    member this.test_is_whitespace() = 
        [|' '; '\t'; '\r'; '\n'; '\u00A0'|]
        |> Array.map (fun c -> Assert.IsTrue(Tokenization.is_whitespace(c))) |> ignore
        [|'A'; '-'|]
        |> Array.map (fun c -> Assert.IsFalse(Tokenization.is_whitespace(c))) |> ignore

    [<TestMethod>]
    member this.test_is_control() = 
        Assert.IsTrue(Tokenization.is_control('\u0005'))
        // TODO how to handle ? '\U0001F4A9'
        [|'A'; ' '; '\t'; '\r'|]
        |> Array.map (fun c -> Assert.IsFalse(Tokenization.is_control(c))) |> ignore

    [<TestMethod>]
    member this.test_is_punctuation() = 
        [|'-'; '$'; '`'; '.'|] |> Array.map (fun c -> Assert.IsTrue(Tokenization.is_punctuation(c))) |> ignore
        [|'A'; ' '|] |> Array.map (fun c -> Assert.IsFalse(Tokenization.is_punctuation(c))) |> ignore
