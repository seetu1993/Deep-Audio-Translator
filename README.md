# NMT



Models explored:
●	Recurrent Neural Networks
●	Recurrent Neural Networks with word embeddings
●	Bidirectional Recurrent Neural Networks
●	Recurrent Neural Networks with word embeddings + Bidirectional Recurrent Neural Networks
●	Encoder – Decoder Model 
BLEU_score.py : Model is loaded and BLEU score is calculated by predicting the translations of the validation set. Corpus bleu api is called with predictions and actual translations as input.
MT_demo : demo for the python application for machine translation 
Folder name : Web_App_Machine_Translation(Seetu)\mt_app 
install the required requirements from requirements.txt
Command to run the Application : 
export FLASK_APP=main
flask run 
Backend of the web application 
main.py : It parses the user input and calls the translate function with the text input and the loaded model as the input argument.
model.py : It contains the translate function which calls model.predict function to predict the translation and return that translation to the main function.
Frontend of the web application
templates folder : contains the HTML code
static folder : contains the css and jss code.
