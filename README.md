# protein-secondary-structure-predictor
## Created for my Introduction to Bioinformatics class. 

Uses sklearn's MLPClassifier and a protein's amino acid sequence to predict its secondary structure. 

## List of Files
 * .gitignore: a gitignore
 * dataset.txt: the 22 sequences and their structures used to train the neural network 
 * predictor.ipynb: a jupyter notebook that runs the program
 * predictor.py: all the python code from the jupyter notebook
 
## Steps to run
 1. Clone the repository
 2. Open and run the predictor.ipynb file using Jupyter Notebook
 3. Create a .txt file with the sequences, follow sample_predict_seq.txt for formatting 
 4. Go to this line in the code: file_predict=open("sample_predict_seq.txt", "r") and replace "sample_predict_seq.txt" <br>
    with the path to your .txt file with the sequences you want to predict
 5. Run the notebook

