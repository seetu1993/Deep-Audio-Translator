import tensorflow as tf
import numpy as np
import model_preprocess

# Load the saved model
model = tf.keras.models.load_model("/home/azureuser/final_model5")
from nltk.translate.bleu_score import corpus_bleu

# calculation of BLEU Score 
predicted_sentences = []
actual_sentences = []
for i in range(len(tmp_x)):
  predicted = logits_to_text(model.predict(tmp_x[i])[0], french_tokenizer)
  predicted_sentences.append(predicted)
  actual_sentences.append(french_sentences[i])

print('Bleu-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('Bleu-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('Bleu-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
print('Bleu-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))