import tensorflow_hub as hub
# import numpy as np

def load_model():
    checkpoint = "https://tfhub.dev/google/universal-sentence-encoder/4"
    #  = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer = hub.load(checkpoint)
    
    return tokenizer


def Encode(inputs):
    tokenizer = load_model()
    out = tokenizer(inputs)
    # print(out.shape)
    return out