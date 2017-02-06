(ns com.lemonodor.lewis.train
  (:require [com.lemonodor.lewis.util :as util]
            [clojure.java.io :as io])
  (:import [opennlp.tools.util ObjectStream MarkableFileInputStreamFactory PlainTextByLineStream TrainingParameters]
           [opennlp.tools.doccat
            BagOfWordsFeatureGenerator
            DoccatFactory
            DocumentCategorizerME]
           [opennlp.tools.namefind NameFinderME NameSampleDataStream TokenNameFinderFactory]
           [opennlp.tools.tokenize SimpleTokenizer]))


(defn training-parameters% [cutoff iterations]
  (let [tp (TrainingParameters.)]
    (.put tp TrainingParameters/CUTOFF_PARAM (str cutoff))
    (.put tp TrainingParameters/ITERATIONS_PARAM (str iterations))
    tp))


(defn write-intent-samples-to-file [samples file]
  (with-open [o (io/writer file :encoding "UTF-8")]
    (doseq [[category examples] samples]
      (doseq [example examples]
        (.write o (format "%s\t%s\n" category example))))))


(defn train-intents [samples cutoff iterations]
  (util/with-temp-file [training-file "intent-training" "txt"]
    (write-intent-samples-to-file samples training-file)
    (let [ds (com.lemonodor.lewis.train.CategoryDataStream. (into-array [training-file]) nil)
          bowfg (BagOfWordsFeatureGenerator.)
          tp (training-parameters% cutoff iterations)
          df (DoccatFactory. SimpleTokenizer/INSTANCE (into-array [bowfg]))
          model (DocumentCategorizerME/train "en" ds tp df)]
      model)))


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
        (com.lemonodor.lewis.train/train-intents
         [["WhereTaxi" ["How far away is my taxi?"
                        "How far away is my cab?"]]
          ["Help" ["Help" "Help me"]]]
         1
         100))

#_(com.lemonodor.lewis.train/predict-intents "help me" intent-model)


(defn write-entity-samples-to-file [samples file]
  (with-open [o (io/writer file :encoding "UTF-8")]
    (doseq [sample samples]
      (.write o (format "%s\n" sample)))))


(defn train-entity-extractor [samples cutoff iterations]
  (util/with-temp-file [training-file "entity-training" "txt"]
    (write-entity-samples-to-file samples training-file)
    (let [sample-stream (-> (MarkableFileInputStreamFactory. training-file)
                            (PlainTextByLineStream. "UTF-8")
                            (NameSampleDataStream.))]
      (NameFinderME/train "en" "drink" sample-stream
                          (training-parameters% cutoff iterations)
                          (TokenNameFinderFactory.)))))


(defn predict-entities [text model]
  (let [finder (NameFinderME. model)
        tokens (.tokenize SimpleTokenizer/INSTANCE text)
        spans (.find finder tokens)]
    (map (fn [span]
           [(.getStart span) (.getEnd span) (.getProb span)])
         (seq spans))))

#_(def drink-model
    (com.lemonodor.lewis.train/train-entity-extractor
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

#_(com.lemonodor.lewis.train/predict-entities "i'll have a vodka" drink-model)
#_(com.lemonodor.lewis.train/predict-entities "i'll have a martini please" drink-model)
