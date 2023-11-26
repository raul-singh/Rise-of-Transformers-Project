import tensorflow as tf
import numpy as np
import yaml

import tensorflow_hub as hub
import tensorflow_text as text

tfk = tf.keras
tfkl = tf.keras.layers
kb = tf.keras.backend


class EncoderTTA(tfk.Model):
    def __init__(self, encoder, tta_pipeline, tta_n=10, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.tta_pipeline = tta_pipeline
        self.tta_n = tta_n
        
    def call(self, features, training=False):
        return tf.math.reduce_mean(tf.stack([self.encoder(self.tta_pipeline(features)) for _ in range(self.tta_n)], axis=-1), axis=-1)


class CLIP(tfk.Model):
    def __init__(self, image_encoder, text_encoder, **kwargs):
        super().__init__(**kwargs)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.loss_tracker = tfk.metrics.Mean(name="loss")
        self.temp = self.add_weight(
            name='t',
            shape=(1, ),
            initializer=tfk.initializers.Constant(1.),
            trainable=True
        )

        # Necessary for initialization
        self.initialize_model()

        
    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):

        image_emb = self.image_encoder(features["image"], training=training)
        text_emb = self.text_encoder(features["caption"], training=training)

        return image_emb, text_emb
        
        if self.image_tta and not training:
            image_emb = tf.math.reduce_mean(tf.stack([self.image_encoder(self.image_tta(features["image"])) for _ in range(self.tta_n)]))
        else:
            image_emb = self.image_encoder(features["image"], training=training)
        if self.text_tta and not training:
            text_emb = tf.math.reduce_mean(tf.stack([self.text_encoder(tf.stack([self.text_tta({"caption": feature})["caption"] for feature in features["caption"]])) for _ in range(self.tta_n)])) # Pipeline is optimized for data preprocessing, thus ["caption"] must be accessed after call and a reshape is needed
        else:
            text_emb = self.text_encoder(features["caption"], training=training)
        
        

    def CLIP_loss(self, image_emb, text_emb):
        norm_image_emb = tf.math.l2_normalize(image_emb, axis=1)
        norm_text_emb = tf.math.l2_normalize(text_emb, axis=1)

        logits = tf.linalg.matmul(norm_image_emb, norm_text_emb, transpose_b=True) * tf.math.exp(self.temp)

        n = tf.shape(logits)[0]
        labels = tf.range(n)

        labels = tf.one_hot(labels, n)

        loss_img = tfk.losses.categorical_crossentropy(labels, logits, from_logits=True)
        loss_txt = tfk.losses.categorical_crossentropy(labels, kb.transpose(logits), from_logits=True)

        return (loss_img + loss_txt) / tf.constant(2.0)

    def train_step(self, features):
        with tf.GradientTape() as tape:
            image_embeddings, caption_embeddings = self(features, training=True)
            loss = self.CLIP_loss(caption_embeddings, image_embeddings)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        image_embeddings, caption_embeddings = self(features, training=False)
        loss = self.CLIP_loss(caption_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    # Workaround
    def initialize_model(self):

        image = tf.reshape(tf.convert_to_tensor(np.zeros((128,128,3))), (1,128,128,3))
        caption = tf.convert_to_tensor(["Hello there"], dtype=tf.string)

        sample = {"image": image, "caption": caption}

        self(sample)

    def summary(self):
        super().summary()

        print("\n")
        self.image_encoder.summary()

        print("\n")
        self.text_encoder.summary()

    def save_weights(self, path):
        self.trainable = False
        self.compile()
        super().save_weights(path)

    def load_weights(self, path):
        self.trainable = False
        self.compile()
        super().load_weights(path)

    def save_checkpoint(self, path):
        super().save_weights(path)

    def load_checkpoint(self, path):
        super().load_weights(path)


def projection(embedding_input, embed_dim, name):
    
    embeddings = tfkl.Dense(embed_dim, name=f'{name}_1')(embedding_input)
    x = tf.nn.selu(embeddings)
    x = tfkl.Dense(embed_dim, name=f'{name}_2')(x)
    x = tfkl.Dropout(0.1)(x)
    x = tfkl.Add()([x, embeddings])
    embeddings = tfkl.LayerNormalization()(x)

    return embeddings


def image_encoder(input_shape, embed_dim, supernet, preprocessing, seed=42):
    
    tf.random.set_seed(seed)

    input_layer = tfkl.Input(shape=input_shape, name='img_input_layer')
    x = preprocessing(input_layer)
    x = supernet(x)
    x = tfkl.GlobalAveragePooling2D(name='GAP')(x)

    x = projection(x, embed_dim, 'img_embedding_dense_layer')
    
    # Connect input and output through the Model class
    cnn_encoder = tfk.Model(inputs=input_layer, outputs=x, name='image_encoder')

    # Return the encoder
    return cnn_encoder


def text_encoder(embed_dim, preprocess, transformer, trainable=True):

    transformer.trainable = trainable
    
    input_layer = tfkl.Input(shape=(), dtype=tf.string, name="text_input")
    x = preprocess(input_layer)
    x = transformer(x)["pooled_output"]
    x = projection(x, embed_dim, 'txt_embedding_dense_layer')

    text_encoder = tfk.Model(inputs=input_layer, outputs=x, name="text_encoder")
    
    return text_encoder


def build_clip(settings_path, weights_path=None, load_weights=True):

    with open(settings_path, "r") as stream:
        try:
            model = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    text_preprocess, text_transformer, img_preprocess, img_supernet = download_model(model['download'])

    model_settings = model['model']
    image_shape = tuple(model_settings['img_input_shape'])

    print('Building clip...')

    clip_text_encoder = text_encoder(model_settings['embed_dim'], text_preprocess, text_transformer)
    clip_image_encoder = image_encoder(image_shape, model_settings['embed_dim'], img_supernet, img_preprocess)

    clip = CLIP(clip_image_encoder, clip_text_encoder)
    
    clip.compile(
        optimizer = tf.optimizers.AdamW(
            learning_rate=model_settings['learning_rate'], 
            weight_decay=model_settings['weight_decay']
            )
        )

    if load_weights:
        
        print('Loading parameters...')

        weights = model_settings['weights_path'] if weights_path is None else weights_path
        clip.load_weights(weights)

    print('Done.')

    return clip_image_encoder, clip_text_encoder, clip
    


def download_model(download):

    print('Downloading models...')

    text_preprocess = hub.KerasLayer(
        download['text_preprocess'],
        name="text_preprocessing",
    )

    text_transformer = hub.KerasLayer(
        download['text_transformer'],
        trainable=True,
        name="transformer",
    )

    cnn = getattr(tfk.applications, download['img_supernet'])
    cnn_pre = getattr(tfk.applications, download['img_preprocess'])

    img_preprocess = cnn_pre.preprocess_input
    img_supernet = cnn(weights='imagenet', include_top=False)
    
    print('Models downloaded.')

    return text_preprocess, text_transformer, img_preprocess, img_supernet
