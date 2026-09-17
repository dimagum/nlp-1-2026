import re
from gensim.models import FastText
import string
from gensim.models.word2vec import PathLineSentences
import numpy as np
from typing import List, Tuple

# =========================================
# 1. Training initial embeddings (4 points)
# =========================================

# 1.1 Training initial embeddings (clean_wap_sentence)
def clean_wap_sentence(sentence: str) -> str:
    """
    Cleans a sentence by performing the following operations:
    1. Replaces newline characters (`\n`) with a space.
    2. Replaces double hyphens (`--`) with a space.
    3. Removes all punctuation except for apostrophes.
    4. Converts the sentence to lowercase.
    5. Collapse the sequence of whitespace characters into a single space. 
    6. Remove the spaces at the beginning and end of the sentence.
    
    Args:
        sentence (str): The input sentence to be cleaned.

    Returns:
        str: The cleaned sentence.
    """    

    sentence = re.sub(r'\n+', ' ', sentence)   # 1
    sentence = re.sub(r'--+', ' ', sentence)   # 2

    punctuation_without_apostrophy = string.punctuation.replace("'", "")
    sentence = sentence.translate(str.maketrans("", "", punctuation_without_apostrophy))    # 3
    sentence = sentence.lower()    # 4
    sentence = re.sub(r'\s+', ' ', sentence)    # 5
    sentence = sentence.strip()    # 6

    return sentence

# 1.2 Training initial embeddings (model validation)
def build_fasttext_model(wap_sentences: PathLineSentences) -> FastText:
    """
    Create a Gensim FastText model for the preprocessed War and Peace corpus
    and build its vocabulary from the provided tokenized sentences.

    Important: Initialize the FastText model with bucket=100_000

    Args:
        wap_sentences: An iterable of tokenized War and Peace sentences.

    Returns:
        FastText: An initialized FastText model with vocabulary built.
    """

    model = FastText(
        vector_size=100,
        window=5,
        sg=1,
        alpha=3e-2,
        seed=42,
        bucket=100000,
        workers=1
    )

    model.build_vocab(wap_sentences)

    return model

# 1.2 Training initial embeddings (model validation)
def train_fasttext_model(wap_sentences: PathLineSentences, model:FastText) -> FastText:
    """
    Train a Gensim FastText model on the provided corpus iterable.

    Args:
        wap_sentences: An iterable of tokenized sentences used for training.
        model: A FastText model with an initialized vocabulary.
    Returns:
        FastText: The trained FastText model.
    """

    model.train(
        corpus_iterable=wap_sentences,
        total_examples=model.corpus_count,
        epochs=5,
    )

    return model

# =========================================
# 2. Finetune using quora dataset (3 points)
# =========================================

# 2.1 Finetune using quora dataset (preprocess_file)
# Define a function to preprocess a single line
def preprocess_line(line: str) -> str:
    """
    Preprocesses a single line of text by:
    1. Removing all punctuation except apostrophes.
    2. Converting the text to lowercase.
    3. Collapse the sequence of whitespace characters into a single space. 
    4. Remove the spaces at the beginning and end of the sentence.

    Args:
        line (str): The input line of text to preprocess.

    Returns:
        str: The preprocessed line of text.
    """

    punctuation_without_apostrophy = string.punctuation.replace("'", "")
    line = line.translate(str.maketrans("", "", punctuation_without_apostrophy))    # 1
    line = line.lower()    # 2
    line = re.sub(r'\s+', ' ', line)    # 3
    line = line.strip()    # 4

    return line

# 2.1 Finetune using quora dataset (preprocess_file)
# Preprocess the text file line by line
def preprocess_file(input_file_path: str, output_file_path: str) -> int:
    """
    Reads a text file line by line, preprocesses each line,
    and writes the preprocessed lines to a new file.

    Args:
        input_file_path (str): Path to the input text file.
        output_file_path (str): Path to the output text file where preprocessed lines will be saved.

    Returns:
        int: The total number of lines processed in the input file.
    """

    lines_processed = 0

    with open(input_file_path, 'r', encoding="utf-8") as input_file, open(output_file_path, 'w', encoding="utf-8") as output_file:
        for line in input_file:
            processed_line = preprocess_line(line)
            output_file.write(processed_line + "\n")


            lines_processed += 1
    
    return lines_processed

# 2.2 Finetune using quora dataset (model validate)
# Update existing vocabulary with the new sentences
def update_fasttext_vocab(finetuned_model: FastText, sentences: PathLineSentences) -> FastText:
    """
    Update the vocabulary of an existing FastText model with new sentences.

    Args:
        finetuned_model: The FastText model whose vocabulary will be updated.
        sentences: An iterable containing the new tokenized sentences.

    Returns:
        The FastText model with the updated vocabulary.
    """
    
    finetuned_model.build_vocab(sentences, update=True)

    return finetuned_model

# 2.2 Finetune using quora dataset (model validate)
def update_fasttext_model(finetuned_model: FastText, sentences: PathLineSentences) -> FastText:
    """
    Continue training an existing FastText model on new sentences.

    Args:
        finetuned_model: The FastText model to continue training.
        sentences: An iterable containing the tokenized training sentences.

    Returns:
        The additionally trained FastText model.
    """
    
    finetuned_model.train(
        corpus_iterable=sentences,
        total_examples=finetuned_model.corpus_count,
        epochs=1,
        start_alpha=finetuned_model.min_alpha,
        end_alpha=finetuned_model.min_alpha
    )

    return finetuned_model

# =========================================
# 3. Implement a way to search similar quora questions given a new query (3 points)
# =========================================

# 3.1.2 Implement a way to search similar quora questions given a new query (get_text_embedding, normalize_vector)
def normalize_vector(embedding: np.ndarray) -> np.ndarray:
    """
    Normalizes a given vector to have unit length.

    Args:
        embedding (np.ndarray): A NumPy array representing the vector to normalize.

    Returns:
        np.ndarray: A normalized vector with unit length.
    """

    norm = np.linalg.norm(embedding)
    if norm <= 1e-6:
        return embedding

    return embedding / norm

# 3.1.1 Implement a way to search similar quora questions given a new query (get_text_embedding)
# 3.2 Implement a way to search similar quora questions given a new query (create_embeddings_storage, get_text_embedding)
def get_text_embedding(text: str, model: FastText) -> np.ndarray:
    """
    Computes the embedding for a given text using a pre-trained FastText model.
    The embedding is the mean of word embeddings for words in the text, normalized to unit length.

    Args:
        text (str): The input text to embed.
        model (FastText): The FastText model to use for generating embeddings.

    Returns:
        np.ndarray: The normalized embedding for the input text.
                    If no valid words are found in the text, returns a zero vector.
    """
    
    processed_text = preprocess_line(text)
    words = processed_text.split()

    if not words:
        return np.zeros(model.wv.vector_size, dtype=np.float32)

    word_embeddings = [
        model.wv[word] 
        for word in words
    ]

    mean_embedding = np.mean(word_embeddings, axis=0)

    normalized_word_embedding = normalize_vector(mean_embedding)

    return normalized_word_embedding

# 3.2 Implement a way to search similar quora questions given a new query (create_embeddings_storage, get_text_embedding)
def create_embeddings_storage(quora_file: str, model: FastText) -> np.ndarray:
    """
    Creates a NumPy array of normalized embeddings for all questions in a dataset.

    Each line in the input file represents a question, which is preprocessed using
    'preprocess_line' function and embedded using 'get_text_embedding' function.

    Args:
        quora_file (str): Path to the file containing the preprocessed questions, one per line.
        model (FastText): The FastText model used to generate embeddings.

    Returns:
        np.ndarray: A 2D NumPy array where each row corresponds to the normalized embedding
                    of a question in the dataset. Shape: (num_questions, embedding_dim).
    """
    embeddings_list: List[np.ndarray] = []
    
    with open(quora_file, "r", encoding="utf-8") as file:
        for line in file:
            processed_line = preprocess_line(line)
            line_embedding = get_text_embedding(text=processed_line, model=model)
            embeddings_list.append(line_embedding)

    # Convert the list of embeddings to a NumPy array
    return np.array(embeddings_list)

# 3.3 Implement a way to search similar quora questions given a new query (find_closest_match_np)
def find_closest_match_np(query: str, model: FastText, embeddings_storage: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
    """
    Finds the closest match(es) to a new question in the database using cosine similarity.

    This function preprocesses the input question, calculates its embedding, and then computes
    cosine similarities between the new question's embedding and a pre-existing database of embeddings.
    It returns the indices of the top-k most similar questions along with their similarity scores.

    Parameters:
        query (str): The input question to be compared.
        model: The trained FastText model used to generate word embeddings.
        embeddings_storage (np.ndarray): A numpy array containing the precomputed and normalized embeddings of all questions in the database.
        k (int, optional): The number of closest matches to return. Defaults to 1.

    Returns:
        Tuple[List[int], List[float]]:
            - List[int]: A list of indices of the top-k most similar questions in the database.
            - List[float]: A list of similarity scores corresponding to these top-k matches.
    """

    # Preprocess the query and embed the question
    processed_query = preprocess_line(query)
    query_embedding = get_text_embedding(text=processed_query, model=model)

    # Compute cosine similarity
    # Dot product if the vectors are normalized embeddings
    # You might want to look up np.dot
    similarities = np.dot(embeddings_storage, query_embedding)

    # Get the indices of the top-k most similar questions
    # You might want to look up np.argsort

    top_k_indices, top_k_similarities = np.argsort(similarities)[-k:][::-1], similarities[np.argsort(similarities)[-k:][::-1]]
    return top_k_indices, top_k_similarities
