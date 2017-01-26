(ns com.lemonodor.lewis.train
  (:require [com.lemonodor.lewis.util :as util]
            [clojure.java.io :as io])
  (:import [opennlp.tools.util ObjectStream TrainingParameters]
           [opennlp.tools.doccat
            BagOfWordsFeatureGenerator
            DoccatFactory
            DocumentCategorizerME]
           [opennlp.tools.tokenize SimpleTokenizer]))


(defn training-parameters% [cutoff iterations]
  (let [tp (TrainingParameters.)]
    (.put tp TrainingParameters/CUTOFF_PARAM (str cutoff))
    (.put tp TrainingParameters/ITERATIONS_PARAM (str iterations))
    tp))


(defn write-samples-to-file [samples file]
  (with-open [o (io/writer file :encoding "UTF-8")]
    (doseq [[category examples] samples]
      (doseq [example examples]
        (.write o (format "%s\t%s\n" category example))))))


(defn train-intents [samples cutoff iterations]
  (util/with-temp-file [training-file "training" "txt"]
    (print training-file)
    (write-samples-to-file samples training-file)
    (let [ds (com.lemonodor.lewis.train.CategoryDataStream. (into-array [training-file]) nil)
          bowfg (BagOfWordsFeatureGenerator.)
          tp (training-parameters% cutoff iterations)
          df (DoccatFactory. SimpleTokenizer/INSTANCE (into-array [bowfg]))
          model (DocumentCategorizerME/train "en" ds tp df)]
      model)))


(defn predict-intents [text model]
  (let [categorizer (DocumentCategorizerME. model (into-array [(BagOfWordsFeatureGenerator.)]))
        tokens (.tokenize SimpleTokenizer/INSTANCE text)
        probs (.categorize categorizer tokens)]
    (println probs)
    (println (type probs))
    (sort-by second
             >
             (map (fn [i]
                    [(.getCategory categorizer i) (aget probs i)])
                  (range (.getNumberOfCategories categorizer))))))
