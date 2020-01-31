// Apache 2.0 from https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
(*
# Copyright 2019 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*)

(*
# Predicting Movie Review Sentiment with BERT on TF Hub

If you’ve been following Natural Language Processing over the past year, you’ve probably heard of BERT: Bidirectional Encoder Representations from Transformers. It’s a neural network architecture designed by Google researchers that’s totally transformed what’s state-of-the-art for NLP tasks, like text classification, translation, summarization, and question answering.

Now that BERT's been added to TF Hub as a loadable module, it's easy(ish) to add into existing Tensorflow text pipelines. In an existing pipeline, BERT can replace text embedding layers like ELMO and GloVE. Alternatively, finetuning BERT can provide both an accuracy boost and faster training time in many cases.

Here, we'll train a model to predict whether an IMDB movie review is positive or negative using BERT in Tensorflow with tf hub. Some code was adapted from this colab notebook. Let's get started!
*)
// TODO get

#if NOTEBOOK
// TODO #r nuget
#else
// TODO #r packet
#endif

//#load "run_classifier.fs"
//#load "optimization.fs"
//#load "tokenization.fs"

//# Download and process the dataset files.
