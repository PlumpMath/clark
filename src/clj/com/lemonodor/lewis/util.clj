(ns com.lemonodor.lewis.util
)


(defn temp-file [prefix suffix]
  (java.io.File/createTempFile prefix suffix))


(defmacro with-temp-file
  [[f prefix suffix] & body]
  `(let [~f (java.io.File/createTempFile ~prefix ~suffix)]
     (try
       ~@body
       (finally
         (.delete ~f)))))
