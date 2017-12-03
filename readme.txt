The code has been executed on CADE machine 12/02/2017 Lab1-12 and uses python 3.6
Execution directory --- u0867999@lab1-12:~/Documents/MLHW5

This directory contains -

1. perceptron.py contains the python code to compile the files.
2. Shell script run.sh to run the program for all 6 modes.
3. Data forder contains raw input.
4. Output folder contains all output
5. hw.pdf contains the pdf of the homework.
6. moreData containt the tree data.
7. readme.txt which contains instructions on how to run the program.

WARNING: Whole code may take a whole day or more to run. 

To compile and execute the program, execute the run.sh script.
./run.sh

or

To execute the program, use the following command while in the perceptron directory
/usr/local/bin/python3.6 perceptron.py


WARNING: Whole code may take a whole day or more to run. To run each thing separately, you can run these commands

SVM: /usr/local/bin/python3.6 perceptron.py 5 6
LogReg: /usr/local/bin/python3.6 perceptron.py 6 7
Naive Bayes: /usr/local/bin/python3.6 perceptron.py 7 8
Bagged Forest: /usr/local/bin/python3.6 perceptron.py 8 9
SVM over Trees: /usr/local/bin/python3.6 perceptron.py 9 10
LogReg over Trees: /usr/local/bin/python3.6 perceptron.py 10 11

Or you can reduce the amount of epochs of cross validation by changing the value of variable "ep" in the "crossVal3" function in perceptron.py. 
