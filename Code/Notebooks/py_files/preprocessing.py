import random
import numpy as np
import pandas as pd

import tensorflow as tf

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Split dataset into train/test/validation
def split(x, test_size=0.2, val_size=0.0, seed=0):
    if val_size + test_size >= 1:
        return None
    x_train, x_test = train_test_split(
        x, test_size=test_size + val_size, random_state=seed
    )
    x_val = None
    if val_size > 0:
        x_test, x_val = train_test_split(
            x_test,
            test_size=val_size / (test_size + val_size),
            random_state=seed,
        )
    return x_train, x_val, x_test

# Loads an image given its path
def load_image_from_path(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3, dct_method="INTEGER_ACCURATE")

    # may need resizing
    #image = tf.image.resize(image, image_shape[:2])
    image = tf.cast(image, dtype=tf.float16)
    image = image / 255.0
    return image

# Loads all features of the given CLIP dataset given the image folder, captions/concepts files and a concept encoder
def load_features(image_folder, captions_file, concepts_file, concept_encoder, filter_percent=1):
    features = []
    
    # Import CSVs
    csv_caption_dataset = tf.data.experimental.CsvDataset(
        captions_file,
        field_delim='\t',
        record_defaults=[tf.string, tf.string],
        header=True,
        select_cols=[0, 1]
    )
    csv_concept_dataset = tf.data.experimental.CsvDataset(
        concepts_file,
        field_delim='\t',
        record_defaults=[tf.string, tf.string],
        header=True,
        select_cols=[0, 1]
    )
    
    # We make the assumption that CSV files contain the same key values (image names)
    # following the same ordering

    # Extract features from dataset
    print("Extracting features from CSV file(s)")
    for caption_el, concept_el in tqdm(zip(csv_caption_dataset, csv_concept_dataset)):
        filename_cap, caption = caption_el
        filename_con , concepts = concept_el
        
        # Sanity check
        assert filename_cap == filename_con
        
        image_path = image_folder + "/" + filename_cap + ".jpg"
        
        features.append({
            'caption': caption,
            'image path': image_path,
            'concepts': concept_encoder.transform([concepts.numpy().decode("utf-8").split(";")]),
        })
        
    # Filter elements
    if filter_percent != 1:
        n_features = int(len(features) * filter_percent)
        features = random.sample(features, n_features)
        
    return features

# Manual preprocessing of features
def preprocess_features(features, concept_encoder, filter_percent=1):
    print("Preprocessing features")
    
    # Filter elements
    if filter_percent != 1:
        n_features = int(len(features) * filter_percent)
        features = random.sample(features, n_features)
        
    return {
        'image paths': tf.convert_to_tensor([x["image path"] for x in tqdm(features)], dtype=tf.string),
        'captions': tf.convert_to_tensor([x["caption"] for x in tqdm(features)], dtype=tf.string),
        'concepts': tf.convert_to_tensor(np.vstack([concept_encoder.transform(x["concepts"]).flatten() for x in tqdm(features)]), dtype=tf.bool),
        # 'images': tf.convert_to_tensor([load_image(x["image path"]) for x in tqdm(features)], dtype=tf.float16),
    }

# Creates a tensorflow dataset given a CLIP dataset and other parameters
def create_dataset(
        features,                       # Dataset features to include in the dataset
        input_features_types,           # Dictionary containing the structure of the dataset features
        feature_shapes,                 # Dictionary containing the shapes of the dataset features
        x_features, y_features=None,    # Lists containing the names for the desired x and y features of the dataset
        x_dict=True, y_dict=True,       # Set to True in order to store x and y features into a dictionary format inside the dataset
        load_images=True,               # Set to True in order to add images to the dataset
        shuffle_buffer_size=1024,       # Shuffle buffer size for the dataset, set to 1 for no shuffling
        batch_size=10,                  # Batch size for the dataset
        cached=False                    # Set to True to enable caching
):
    # Generate dataset following initial input feature types
    dataset = tf.data.Dataset.from_generator(
        lambda: features, { x: input_features_types[x] for x in input_features_types }
    )
    
    # Preprocessing internal functions
    def setshape(e):
        for (k, v) in feature_shapes.items():
            if k in e:
                e[k].set_shape(v)
        return e
    def add_images(e):
        # Maybe parametrize
        img_from = "image path"
        img_to = "image"
        new_features = list(input_features_types.keys()) + [img_to]
        return {f:e[f] if f != img_to else load_image_from_path(e[img_from]) for f in new_features}
    def split_xy(e):
        e_x = {xf:tf.squeeze(e[xf]) for xf in x_features} if x_dict else tf.squeeze([e[xf] for xf in x_features])
        if y_features:
            e_y = {yf:tf.squeeze(e[yf]) for yf in y_features} if y_dict else tf.squeeze([e[yf] for yf in y_features])
            return (e_x, e_y)
        return e_x
    
    # Preprocess
    if load_images:
        dataset = dataset.map(add_images)
    dataset = dataset.map(setshape)
    dataset = dataset.map(split_xy)

    # Compile dataset
    if cached:
        dataset = dataset.cache()
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

# Top level function to import datasets
# Returns (concept_list, concept_onehot_encoder), [(train_dataset, val_dataset, test_dataset)], (train_ds_size, val_ds_size, test_ds_size)
def import_datasets(
    image_folder,                   # Folder containing images
    captions_file,                  # File path for the captions file
    concepts_file,                  # File path for the concepts file (concept id <-> concept text)
    concepts_det_file,              # File path for the concepts detection train file (element id <-> concept id)
    input_features_types,           # Dictionary containing the structure of the dataset features
    feature_shapes,                 # Dictionary containing the shapes of the dataset features
    dataset_parameters,             # List containing dictionaries specifying dataset parameters for each desired output dataset collection, parameters are:
                                    # x_features, y_features:None,      Lists containing the names for the desired x and y features of the dataset
                                    # x_dict:True, y_dict:True,         Set to True in order to store x and y features into a dictionary format inside the dataset
                                    # load_images:True,                 Set to True in order to add images to the dataset
                                    # shuffle_buffer_size:1024,         Shuffle buffer size for the dataset, set to 1 for no shuffling
                                    # batch_size:10,                    Batch size for the dataset
                                    # cached:False,                     Set to True to enable caching
    filter_percent=1,               # Percentage of the dataset to retrieve
    test_size=0.2,                  # Percentage of test dataset
    val_size=0.0,                   # Percentage of validation dataset
    seed=0,                         # Random seed
):
    # Loads concepts
    concepts = pd.read_csv(concepts_file, sep='\t')
    concept_list = concepts.set_index('concept')['concept_name'].to_dict()
    # Concept one-hot encoder
    concepts_onehot = MultiLabelBinarizer(classes=list(concept_list.keys()))
    _ = concepts_onehot.fit([list(concept_list.keys())])

    # Load dataset features from csv files, split them and preprocess them
    features = load_features(image_folder, captions_file, concepts_det_file, concepts_onehot, filter_percent=filter_percent)
    feat_train, feat_val, feat_test = split(features, test_size=test_size, val_size=val_size, seed=seed)

    # Calculate dataset sizes
    train_ds_size = len(feat_train) if feat_train else 0
    val_ds_size = len(feat_val) if feat_val else 0
    test_ds_size = len(feat_test) if feat_test else 0 

    datasets = []
    # Created datasets
    for parameters in dataset_parameters:
        train_dataset = create_dataset(feat_train, input_features_types, feature_shapes, **parameters) if feat_train else None
        val_dataset = create_dataset(feat_val, input_features_types, feature_shapes, **parameters) if feat_val else None
        test_dataset = create_dataset(feat_test, input_features_types, feature_shapes, **parameters) if feat_test else None
        datasets.append((train_dataset, val_dataset, test_dataset))

    return (concept_list, concepts_onehot), datasets, (train_ds_size, val_ds_size, test_ds_size)