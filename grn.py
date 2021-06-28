import numpy as np
import pandas as pd
import re
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNGRU, CuDNNLSTM, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.wrappers import Bidirectional
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from keras import optimizers

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


EMBEDDING_FILES = [
    '../input/glove6b300dtxt/glove.6B.300d.txt',
    # '../input/glove840b300dchar/glove.840B.300d-char.txt'
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
]


NUM_MODELS = 2
BATCH_SIZE = 512
GRN_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * GRN_UNITS
EPOCHS = 4
MAX_LEN = 220
DROP_OUT = 0.3


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except (KeyError, UnicodeDecodeError, UnicodeEncodeError) as e:
            pass
    return embedding_matrix
    

def build_model(embedding_matrix, num_aux_targets):
    input_s = Input(shape=(MAX_LEN,))
    grn_s = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(input_s)
    grn_s = Bidirectional(CuDNNLSTM(GRN_UNITS, return_sequences=True))(grn_s)
    grn_s = Bidirectional(CuDNNLSTM(GRN_UNITS, return_sequences=True))(grn_s)
    grn_s = SpatialDropout1D(0.3)(grn_s)

    hidden = concatenate([
        GlobalMaxPooling1D()(grn_s),
        GlobalAveragePooling1D()(grn_s),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=input_s, outputs=[result, aux_result])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
    

def preprocess(text):
    text = re.sub(r"<[^>]>", '', text)
    emoticons = re.findall(r"(?:|;|=)(?:-)?(?:\)\(|D|P)", text)
    text = re.sub(r"[\W]+", " ", text.lower()) + " ".join(emoticons).replace('-', '')
    text = re.sub(r"\n", ' ', text)
    return text.lower()




train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

train = train.fillna(0.0)
x_train = train['comment_text'].apply(preprocess)
y_train = np.where(train['target'] >= 0.5, 1, 0)
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
x_test = test['comment_text'].apply(preprocess)

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
    
checkpoint_predictions = []
weights = []

for model_idx in range(NUM_MODELS):
    model = build_model(embedding_matrix, y_aux_train.shape[-1])
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            [y_train, y_aux_train],
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=2,
            callbacks=[
                LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))
            ]
        )
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        weights.append(2 ** global_epoch)

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': predictions
})


submission.to_csv('submission.csv', index=False)