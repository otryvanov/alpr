cat test.lst | grep -Eo '.*jpg' | xargs -n 1 bash -c 'echo -n $0 ""; python detect.py test/$0'