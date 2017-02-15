(ns com.lemonodor.clark-test
  (:require [clojure.test :refer :all]
            [com.lemonodor.clark :as clark]))


(deftest intent-classification-test
  (testing "intent classification"
    (let [examples
          [["WHERE-TAXI" ["how far away is my taxi"
                          "how far away is my cab"
                          "can you tell me where my taxi is"]]
           ["WHAT-TIME" ["what time is it"
                         "can you tell me what time it is"]]
           ["HELP" ["Help" "Help me"]]]
          model (clark/train-intents examples)]
      (let [[intent score]
            (first
             (clark/predict-intents "where is my taxi" model))]
        (is (= intent "WHERE-TAXI"))
        (is (number? score)))
      (let [[intent score]
            (first
             (clark/predict-intents "what time are you" model))]
        (is (= intent "WHAT-TIME"))
        (is (number? score))))))
