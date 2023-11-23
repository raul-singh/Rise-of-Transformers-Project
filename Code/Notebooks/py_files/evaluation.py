import os

import numpy as np
import pandas as pd

import tensorflow as tf

from tqdm import tqdm

class EvalMetrics:
    METRIC_ACCURACY = "Accuracy"
    METRIC_MAP = "MAP"
    METRIC_MAR = "MAR"
    METRIC_F1 = "F1"

# Construct reference dataset for retrieving side data of elements
# Unusable due to TensorFlow funny stuff
def generate_dataset_reference(
    dataset_eval,                 # Dataset to generate embeddings
    dataset_ref_map=lambda *x: x, # Lambda mapping function for reference
):
    return [e for e in dataset_eval.map(dataset_ref_map).unbatch()]

# Generate the embeddings and the corresponding dataset reference for an image dataset
def generate_image_embeddings(
    image_encoder,                 # Image encoder of clip model
    dataset_eval,                  # Dataset to generate embeddings (WARNING: the dataset must not be shuffling or have a shuffle buffer size of 1)
    dataset_pred_map=lambda *x: x, # Lambda mapping function for prediction
    dataset_ref_map=lambda *x: x,  # Lambda mapping function for reference
):
    print("Generating image embeddings")
    # Generate image embedding
    image_embeddings = image_encoder.predict(
        dataset_eval.map(dataset_pred_map),
        verbose=1,
    )
    # Construct reference dataset for retrieving side data of elements
    dataset_reference = [e for e in dataset_eval.map(dataset_ref_map).unbatch()]
    return dataset_reference, image_embeddings

# Generate the embeddings and the corresponding dataset reference for a text dataset
def generate_text_embeddings(
    text_encoder,                  # Image encoder of clip model
    dataset_eval,                  # Dataset to generate embeddings (WARNING: the dataset must not be shuffling or have a shuffle buffer size of 1)
    dataset_pred_map=lambda *x: x, # Lambda mapping function for prediction
    dataset_ref_map=lambda *x: x,  # Lambda mapping function for referenc
    is_list=False
):
    print("Generating text embeddings")
    # Generate text embedding
    if is_list:
        # Generate text embedding
        text_embeddings = text_encoder.predict(
            dataset_eval,
            verbose=1,
        )

        dataset_reference = None

    else:
        # Generate text embedding
        text_embeddings = text_encoder.predict(
            dataset_eval.map(dataset_pred_map),
            verbose=1,
        )

        # Construct reference dataset for retrieving side data of elements
        dataset_reference = [e for e in dataset_eval.map(dataset_ref_map).unbatch()]
    return dataset_reference, text_embeddings


# Return the results in the form of reference dataset indexes of a text to image retrieval for a series of queries
def find_t2i_matches(
    queries,                # Queries to search
    text_encoder,           # Text encoder of clip model
    image_embeddings,       # Generated image embeddings
    k=10,                   # Number of elements for top-k
    normalize=True,         # Embedding normalization
):
    print("Computing Text-to-Image matches")
    query_embedding = text_encoder.predict(queries)
    # Normalize the query and the image embeddings
    if normalize:
        image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    # Compute the dot product between the query and the image embeddings
    dot_similarity = tf.matmul(query_embedding, image_embeddings, transpose_b=True)
    # Retrieve top k indices
    results = tf.math.top_k(dot_similarity, k).indices.numpy()
    return results

# Return the results in the form of reference dataset indexes of a image to text retrieval for a series of queries
def find_i2t_matches(
    queries,                # Queries to search
    image_encoder,          # Text encoder of clip model
    text_embeddings,        # Generated image embeddings
    k=10,                   # Number of elements for top-k
    normalize=True,         # Embedding normalization
):
    print("Computing Image-to-Text matches")
    query_embedding = image_encoder.predict(queries)
    # Normalize the query and the text embeddings
    if normalize:
        text_embeddings = tf.math.l2_normalize(text_embeddings, axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    # Compute the dot product between the query and the text embeddings
    dot_similarity = tf.matmul(query_embedding, text_embeddings, transpose_b=True)
    # Retrieve top k indices
    results = tf.math.top_k(dot_similarity, k).indices.numpy()
    return results

# Extract the reference dataset objects given a list of indexes
def index_to_reference(results, dataset_reference):
    return [[dataset_reference[match] | {"index": match} for match in result] for result in results]

# Transform a one-hot encoded list of boolean concepts to the respective list of raw concepts labels
# If the flag string_form is set to true, the returned list will contain strings of concatenated concept text
def decode_concepts(concepts, encoder, concept_list, string_form=True):
    c = np.array(concepts)
    c = encoder.inverse_transform(c)
    if string_form:
        c = [" ".join([concept_list[concept] for concept in e]) for e in c]
    return c


# Retrieve relevant items given a list of queries (DO NOT RUN THIS ON A COMPLETE DATASET!!!)
def retrieve_relevant(queries, dataset_reference, reference_preprocess=lambda x: x, relevance=lambda m, o: m == o):
    return [
        [element for element in map(reference_preprocess, dataset_reference) if relevance(query, element)]
        for query in queries
    ]

# Compute the number of relevant items in the first k matches in a list of results
# If queries is None, it is assumed that the queries ran to obtain the list of results are parallel to the elements in dataset_reference
def compute_relevant_at_k(results, dataset_reference, queries=None, k=None, reference_preprocess=lambda x: x, relevance=lambda m, o: m == o):
    if not k:
        k = len(results[0])
    if queries:
        relevant_reference = retrieve_relevant(queries, dataset_reference, reference_preprocess=reference_preprocess, relevance=relevance)
    else:
        relevant_reference = map(reference_preprocess, dataset_reference)
    return [ 
        np.count_nonzero([relevance(match, reference) for match in list(map(reference_preprocess, matches))[0:k]])
        for matches, reference in zip(results, relevant_reference) 
    ]

# Computes the total number of relevant elements for a dataset or queries
# It is assumed that the element returned by reference_preprocess is hashable and can be used as a dictionary key
# If queries is None, it is assumed that the queries ran to obtain the list of results are parallel to the elements in dataset_reference
def compute_total_relevance(
    dataset_reference, queries=None,                                   # Dataset reference or queries to compute total relevance for
    reference_preprocess=lambda x: x, relevance=lambda m, o: m == o,   # Preprocessing function and relevance function
    load_from_file=True, save_to_file=True,                            # Load/Save flags
    fileinfo={}                                                        # Info for loading/saving data from/to a file in the form of a dictionary with the following keys:
                                                                       # path, filename, test_split, val_split, split, metric, other
                                                                       # if a filename is specified, only the base path is needed
):
    tot_relevant = True
    # Check if queries are passed, if so run general function without loading/saving to file
    if queries:
        relevant_reference = retrieve_relevant(queries, dataset_reference, reference_preprocess=reference_preprocess, relevance=relevance)
        return [len(e) for e in relevant_reference]
    # Check for existing file and load relevance data
    if load_from_file:
        tot_relevant = load_relevance_from_csv(fileinfo)
        if not tot_relevant:
            print("Proceeding with total relevance calculation...")
    if not tot_relevant:
        # Build preprocessed dataset
        relevant_reference = list(map(reference_preprocess, dataset_reference))
        total_n = {}
        # Iterate through dataset and count equal items
        for element in relevant_reference:
            if element in total_n:
                total_n[element] += 1
            else:
                total_n[element] = 1
        # Check bytecode of relevance function to determine if the relevance function is equality,
        # if so, return counts, otherwise apply relevance to the whole dataset
        if not relevance.__code__.co_code == (lambda m, o: m == o).__code__.co_code:
            total_n = {element: sum([total_n[x] for x in total_n if relevance(x, element) and element != x]) + 1 for element in tqdm(total_n)} 
        tot_relevant = [total_n[element] for element in relevant_reference]
    # Check for existing file and save relevance data
    if save_to_file:
        save_relevance_to_csv(tot_relevant, fileinfo)
    return tot_relevant

# Build the filename for a relevance file given some dataset and preprocessing attributes
def build_relevance_filename(
    path,                              # Base path for the file
    filename=None,                     # Name of the csv file to load, if None it will be inferred from dataset attributes
    test_split=0.2,                    # Test split percentage
    val_split=0,                       # Validation split percentage
    split="train",                     # Either "train", "test" or "val"
    metric="",                         # Metric used to compute relevance
    other=[],                          # Other attributes as an ordered list of (name, value) tuples
):
    if not filename:
        filename = "TotRelevant_" + str(test_split) + "_" + str(val_split) + "_" + split + "_" + metric
        for attr in other:
            filename += "_" + attr[0] + "-" + str(attr[1])
        filename += ".csv"
    filename = path + filename
    return filename

# Load a csv relevance file given a filename or some dataset and preprocessing attributes
def load_relevance_from_csv(fileinfo={}):
    filename = build_relevance_filename(**fileinfo)
    if not os.path.exists(filename):
        print(f"The relevance file \"{filename}\" does not exist!")
        return False
    else:
        try:
            return np.squeeze(pd.read_csv(filename, header=None).values.tolist())
        except OSError as error:
            print(f"Couldn't load file \"{filename}\": {error}")
    return False

# Save total relevant data to a csv relevance file given a filename or some dataset and preprocessing attributes
def save_relevance_to_csv(tot_relevant, fileinfo={}):
    filename = build_relevance_filename(**fileinfo)
    if os.path.exists(filename):
        print(f"Overwriting \"{filename}\" relevance file!")
    df = pd.DataFrame(tot_relevant)
    try:
        df.to_csv(filename, index=False, header=False)
        return True
    except OSError as error:
            print(f"Couldn't save file \"{filename}\": {error}")
    return False


def compute_top_k_accuracy(results, dataset_reference, relevant_at_k):
    hits = np.count_nonzero(relevant_at_k)
    return hits / len(dataset_reference)

def compute_map_k(results, dataset_reference, relevant_at_k, k=None):
    if not k:
        k = len(results[0])
    precision_at_k = [r/k for r in relevant_at_k]
    return np.sum(precision_at_k) / len(dataset_reference)

def compute_mar_k(results, dataset_reference, relevant_at_k, total_relevant):
    recall_at_k = [rk/tr for rk, tr in zip(relevant_at_k, total_relevant)]
    return np.sum(recall_at_k) / len(dataset_reference)

def compute_F1_k(precision=0, recall=0):
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
