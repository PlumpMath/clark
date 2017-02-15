(ns com.lemonodor.clark
  (:require
   [com.lemonodor.clark.util :as util]
   [clojure.java.io :as io]
   [clojure.tools.logging :as log])
  (:import
   [com.lemonodor.clark CategoryDataStream]
   [opennlp.tools.util
    Span
    TrainingParameters]
   [opennlp.tools.doccat
    BagOfWordsFeatureGenerator
    DoccatFactory
    DocumentCategorizerME]
   [opennlp.tools.namefind
    NameFinderME
    NameSampleDataStream
    TokenNameFinderFactory]
   [opennlp.tools.tokenize SimpleTokenizer]))

(set! *warn-on-reflection* true)


(def default-training-parameters
  {:cutoff 0 :iterations 100})


(defn training-parameters% [params]
  (let [params (merge default-training-parameters params)
        tp (TrainingParameters.)]
    (.put tp TrainingParameters/CUTOFF_PARAM (str (:cutoff params)))
    (.put tp TrainingParameters/ITERATIONS_PARAM (str (:iterations params)))
    tp))


(defn write-intent-samples-to-file [samples file]
  (with-open [o (io/writer file :encoding "UTF-8")]
    (doseq [[category examples] samples]
      (doseq [example examples]
        (.write o (format "%s\t%s\n" category example))))))


(defn train-intents-file
  ([training-file]
   (train-intents-file training-file {}))
  ([training-file params]
   (log/info "Training on" training-file)
   (let [^CategoryDataStream ds (CategoryDataStream. ^"[Ljava.io.File;" (into-array [(io/file training-file)]) nil)
         ^BagOfWordsFeatureGenerator bowfg (BagOfWordsFeatureGenerator.)
         ^TrainingParameters tp (training-parameters% params)
         ^DoccatFactory df (DoccatFactory. SimpleTokenizer/INSTANCE (into-array [bowfg]))
         model (DocumentCategorizerME/train "en" ds tp df)]
     model)))


(defn train-intents
  ([samples]
   (train-intents samples {}))
  ([samples params]
   (util/with-temp-file [training-file "intent-training" "txt"]
     (write-intent-samples-to-file samples training-file)
     (train-intents-file training-file params))))


(defn predict-intents [text model]
  (let [categorizer (DocumentCategorizerME. model (into-array [(BagOfWordsFeatureGenerator.)]))
        tokens (.tokenize SimpleTokenizer/INSTANCE text)
        probs (.categorize categorizer tokens)
        predictions
        (sort-by second
                 >
                 (map (fn [i]
                        [(.getCategory categorizer i) (aget probs i)])
                      (range (.getNumberOfCategories categorizer))))]
    predictions))

#_(def intent-model
        (com.lemonodor.clark.train/train-intents
         [["WhereTaxi" ["How far away is my taxi?"
                        "How far away is my cab?"]]
          ["Help" ["Help" "Help me"]]]
         1
         100))

#_(com.lemonodor.clark.train/predict-intents "help me" intent-model)


(defn write-entity-samples-to-file [samples file]
  (with-open [o (io/writer file :encoding "UTF-8")]
    (doseq [sample samples]
      (.write o (format "%s\n" sample)))))


(defn train-entity-extractor [samples cutoff iterations]
  (util/with-temp-file [training-file "entity-training" "txt"]
    (write-entity-samples-to-file samples training-file)
    (let [sample-stream (-> (opennlp.tools.util.MarkableFileInputStreamFactory. training-file)
                            (opennlp.tools.util.PlainTextByLineStream. "UTF-8")
                            (NameSampleDataStream.))]
      (NameFinderME/train "en" "drink" sample-stream
                          (training-parameters% cutoff iterations)
                          (TokenNameFinderFactory.)))))


(defn predict-entities [^String text ^opennlp.tools.namefind.TokenNameFinderModel model]
  (let [finder (NameFinderME. model)
        tokens (.tokenize SimpleTokenizer/INSTANCE text)
        spans (.find finder tokens)]
    (map (fn [^Span span]
           [(.getStart span) (.getEnd span) (.getProb span)])
         (seq spans))))

#_(def drink-model
    (com.lemonodor.clark.train/train-entity-extractor
     ["give me a <START:drink> beer <END>"
      "can i have a <START:drink> beer <END>"
      "i ' ll have a <START:drink> beer <END> please"
      "how about a <START:drink> beer <END>"
      "give me a <START:drink> vodka <END>"
      "can i have a <START:drink> vodka <END>"
      "i ' ll have a <START:drink> vodka <END> please"
      "how about a <START:drink> vodka <END>"
      "give me a <START:drink> double scotch <END>"
      "can i have a <START:drink> double scotch <END>"
      "i ' ll have a <START:drink> double scotch <END> please"
      "how about a <START:drink> double scotch <END>"]
     1
     100))

#_(com.lemonodor.clark.train/predict-entities "i'll have a vodka" drink-model)
#_(com.lemonodor.clark.train/predict-entities "i'll have a martini please" drink-model)
