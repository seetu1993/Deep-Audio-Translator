
import collections

import helper
import numpy as np


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy



import os


def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

english_sentences = load_data(r'/home/azureuser/small_vocab_en.csv')
french_sentences = load_data(r'/home/azureuser/small_vocab_fr.csv')

english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(x)
    t=tokenizer.texts_to_sequences(x)
    # TODO: Implement
    return t, tokenizer


# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    padding=pad_sequences(x,padding='post',maxlen=length)
    return padding


# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))

def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

print(preproc_english_sentences[:1])

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

from keras.layers import Input
from keras.models import Sequential

def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Hyperparameters
    learning_rate = 1e-3
    embedding_size = 300
    GRU_units = 512
    FCL_units = french_vocab_size*4
    global epochs
    global batch_size
    epochs = 50
    batch_size = 128

    # Making the model
    # 1. Input sequence
    input_sequence = Input(shape = input_shape[1:])
    
    # 2. Adding the 1st layer - GRU Layer
    encoder = GRU(units = GRU_units, return_sequences = False)(input_sequence)
    
    # Fiting the fixed-sized 2D output of the encoder to the 3D input of the decoder
    repeat_vector = RepeatVector(output_sequence_length)(encoder)
    
    # 3. Adding the 2st layer - GRU Layer
    decoder = GRU(units = GRU_units, return_sequences = True)(repeat_vector)
    
    # 4. Adding the 3nd layer - Fully Connected Layer
    fully_connected_layer = Dense(units = FCL_units, activation = 'relu')(decoder)
    
    # 5. Adding the output layer
    logits = TimeDistributed(Dense(units = french_vocab_size + 1, activation = "softmax"))(fully_connected_layer)
    
    # Making the model
    model = Model(inputs = input_sequence, outputs = logits)
    
    # Compiling the model
    model.compile(loss = sparse_categorical_crossentropy,
                  optimizer = Adam(learning_rate),
                  metrics = ['accuracy'])
    return model

    
# Reshaping the input
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Initializing the model
encoded_decoded_model = encdec_model(input_shape = tmp_x.shape,
                                     output_sequence_length = max_french_sequence_length,
                                     english_vocab_size = len(english_tokenizer.word_index) + 1,
                                     french_vocab_size = len(french_tokenizer.word_index) + 1)

# Training the model
encoded_decoded_model.fit(tmp_x, preproc_french_sentences, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

# Print prediction(s)
print(logits_to_text(encoded_decoded_model.predict(tmp_x[:1])[0], french_tokenizer))

from nltk.translate.bleu_score import corpus_bleu
predicted_sentences = []
actual_sentences = []
for i in range(len(tmp_x)):
  predicted = logits_to_text(encoded_decoded_model.predict(tmp_x[i])[0], french_tokenizer)
  predicted_sentences.append(predicted)
  actual_sentences.append(french_sentences[i])

print('Bleu-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('Bleu-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('Bleu-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
print('Bleu-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))