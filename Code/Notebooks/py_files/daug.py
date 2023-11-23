import tensorflow as tf

class RandomSwapWords(tfkl.Layer):
    def __init__(self, p=0.5, seed=None):
        super(RandomSwapWords, self).__init__()
        self.p = p
        self.seed = seed
   
    def call(self, inputs):
        caption = inputs["caption"]
        
        words = tf.strings.split(caption, " ")
        
        if type(words) is tf.RaggedTensor:
            words = words.to_tensor()
        
        # Create random shuffle mask
        shuffle_mask = tf.cast(tf.random.uniform([tf.shape(words)[0]], seed=self.seed) < self.p, tf.int32)
        # Generate shuffled indexes 
        indexes = tf.range(tf.shape(words)[0])
        shuffled_indexes = tf.boolean_mask(tf.random.shuffle(indexes, seed=self.seed), shuffle_mask)
        original_indexes = tf.where(shuffle_mask > 0)

        # Swap words at shuffled_indexes
        words = tf.tensor_scatter_nd_update(words, original_indexes, tf.gather(words, shuffled_indexes))
        
        return dict(inputs, caption=tf.strings.reduce_join(words, separator=" "))
    
class RandomConceptWords(tfkl.Layer):
    def __init__(self, concept_encoder_classes, concept_list, p=0.5, seed=None):
        super(RandomConceptWords, self).__init__()
        self.classes = tf.constant(concept_encoder_classes, dtype=tf.string)
        self.concept_list_k = tf.constant(list(concept_list.keys()))
        self.concept_list_v = tf.constant(list(concept_list.values()))
        self.p = p
        self.seed = seed
        
    def call(self, inputs):
        caption = inputs["caption"]
        concepts = inputs["concepts"]
        
        words = tf.strings.split(caption, " ")
        
        # Preprocess concepts to obtain concept text
        concepts = tf.where(concepts)
        concepts = tf.gather(self.classes, concepts)
        conceptstext_mask = tf.reduce_any(tf.equal(self.concept_list_k, concepts), axis=0)
        concepts_text = tf.boolean_mask(self.concept_list_v, conceptstext_mask)
        
        # Take a random number of concepts based on p
        concepts_mask = tf.cast(tf.random.uniform([tf.shape(concepts_text)[0]], seed=self.seed) < self.p, tf.int32)
        insert_concepts = tf.boolean_mask(tf.random.shuffle(concepts_text, seed=self.seed), concepts_mask)
        n_insert = tf.reduce_sum(concepts_mask)
        
        # Choose the positions of the concept inserts at random
        # maxval + 2 because top extreme is not included and because we want to be able to add concepts at the end of the caption
        insert_indexes = tf.random.uniform([n_insert], minval=0, maxval=tf.shape(words)[0]+2, dtype=tf.int32)
        
        # Appends (actually prepends) concepts to a caption word if the indexes for the word and the insert position match
        @tf.function
        def append_concepts(word, index, insert_indexes, insert_concepts):
            position_concept_mask = tf.equal(insert_indexes, index)
            # Number of concepts to insert at this index
            insert_flag = tf.reduce_sum(tf.cast(position_concept_mask, tf.int32))
            if insert_flag <= 0:
                return tf.reshape(word, [1])
            word_concepts = tf.boolean_mask(insert_concepts, position_concept_mask)
            if insert_flag > 1:
                # Concatenate concepts before joining them to the caption word if there are multiple
                word_concepts = tf.strings.reduce_join(word_concepts, separator=" ")
            new_word = tf.strings.join([word_concepts, word], separator=" ")
            return tf.reshape(new_word, [1])
        
            
        # Generate indexes to go through caption adding an extra index for the end
        indexes = tf.range(tf.shape(words)[0] + 1)
        words = tf.concat([words, tf.constant([""])], axis=0)
        
        # Append concepts to caption slices (could be optimized by using a vectorized_map but requires workarounds)
        words = tf.map_fn(lambda x: append_concepts(x[0], x[1], insert_indexes, insert_concepts), (words, indexes), fn_output_signature=tf.string)
        
        return dict(inputs, caption=tf.strings.reduce_join(words, separator=" "))