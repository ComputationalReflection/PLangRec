import tensorflow as tf

MODEL_PATH: str = 'models/BRNN'
# Loads the model into memory at startup to go faster upon prediction
model = tf.keras.models.load_model(MODEL_PATH)

def parse_line(line):
    result = list(line) # Split line in characters
    # Tokenization
    result = result[:40] # Truncates line 
    result = [ord(char) for char in result] # Parse characters to its ASCII code
    # Vocabulary
    result = [code if (code >= 32 and code < 127) else 31 for code in result] # Filter vocabulary
    result = [code-30 for code in result] # Makes a consecutive vocabulary
    if len(result) < 40:
        result.extend([0] * (40 - len(result))) # Padding
    return result


def preprocess_char_numbers(ds):
    EMPTY_SPACES = 32 - 2
    ds = ds.map(lambda line: (line - EMPTY_SPACES), num_parallel_calls=2)
    OOV_CHAR = 1
    ds = ds.map(lambda line: (tf.where(line < 32 - EMPTY_SPACES, OOV_CHAR, line)), num_parallel_calls=2)
    ds = ds.map(lambda line: (tf.where(line >= 127 - EMPTY_SPACES, OOV_CHAR, line)), num_parallel_calls=2)
    return ds


def preprocess_numeric_ds(ds):
    ds = ds.map(lambda line: line[:40])
    ds = preprocess_char_numbers(ds)
    ds = ds.map(lambda line: (tf.concat([line, tf.zeros([40 - tf.shape(line)[0]], dtype=tf.dtypes.int32)], axis=0)), num_parallel_calls=2)
    return ds


def predict(line: str) -> dict:
    from configuration import LANGUAGES
    result = {}
    predictions = model.predict([parse_line(line)])
    for i, p in enumerate(predictions[0]):
        result[LANGUAGES[i]] = round(p*100, 2)
    return result
