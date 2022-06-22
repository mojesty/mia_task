from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle
import numpy

from pip import main
import typer

from transformers import AutoTokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def encode(tokenizer: AutoTokenizer, x: str) -> List[str]:
    return tokenizer.convert_ids_to_tokens(tokenizer.encode(x))[1:-1]

def train(
    tokenizer_model: str,
    data: List[str],
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: int = 10_000
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    vectorizer = TfidfVectorizer(
        analyzer=lambda x: encode(tokenizer, x),
        max_features=max_features,
        ngram_range=ngram_range
    )
    vectorizer.fit(data)
    return vectorizer


def train_from_corpus(
    corpus_path: Path,
    output_path: Path,
    tokenizer: str,
):
    df = pd.read_csv(corpus_path, sep=';')
    df['concept_len'] = df.ICD_text.apply(lambda x: len(x.split()))
    df_one_word = df[df.concept_len == 1]
    vectorizer = train(tokenizer, df_one_word.ICD_text.tolist())
    output_path.write_bytes(pickle.dumps(vectorizer))


def index(
    data: List[str],
    vectorizer: TfidfVectorizer
    ):
    index = vectorizer.transform(data)
    return index


def query(
    inputs: str,
    index: numpy.ndarray,
    tokenizer: AutoTokenizer
):
    bpe_query = encode(tokenizer, inputs)
    

if __name__ == '__main__':
    typer.run(train_from_corpus)
