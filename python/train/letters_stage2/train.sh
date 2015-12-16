for LETTER in 'HM' '0O' '1' '2' '3' '4' '5' '6' '7' '8' '8B' '9' 'A' 'B' 'C' 'E' 'H' 'K' 'M' 'P' 'T' 'X' 'Y' ; do
  export LETTER
  python train.py pos_${LETTER} neg_${LETTER}
  mv svm_letter_${LETTER}.xml svm_letter_${LETTER}.xml.old
  mv svm.xml svm_letter_${LETTER}.xml
  mv svm.pickle svm_letter_${LETTER}.pickle
  rm *.pickle
done
