for TYPE in 'vertical_stage1' ; do
  export TYPE
  python train.py pos_${TYPE} neg_${TYPE}
  mv svm.xml svm_${TYPE}.xml
  mv svm.pickle svm_${TYPE}.pickle
done
