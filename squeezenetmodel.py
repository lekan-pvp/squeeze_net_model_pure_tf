from os import name
import tensorflow as tf

class SqueezeNetModel(object):
    # Model Initialization
    def __init__(self, original_dim, resize_dim, output_size):
        self.original_dim = original_dim
        self.resize_dim = resize_dim
        self.output_size = output_size

    # Random crop and flip images
    def random_crop_and_flip(self, float_image):
        crop_image = tf.random_crop(float_image, size=[self.resize_dim, self.resize_dim, 3])
        updated_image = tf.image.random_flip_left_right(crop_image)
        return updated_image

    # Data augmentation
    def image_preprocessing(self, data, is_training):
        reshaped_image = tf.reshape(data, [3, self.original_dim, self.original_dim])
        transposed_image = tf.transpose(reshaped_image, [1, 2, 0])
        float_image = tf.cast(transposed_image, tf.float32)
        if is_training:
            updated_image = self.random_crop_and_flip(float_image)
        else:
            updated_image = tf.image.resize_with_crop_or_pad(float_image, self.resize_dim, self.resize_dim)
        standartized_image = tf.image.per_image_standardization(updated_image)
        return standartized_image

    # Convolution layer wrapper
    def custom_conv2d(self, inputs, filters, kernel_size, name):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            activation=tf.nn.relu,
            padding='same',
            name=name)


    # SqueezeNet fire module
    def fire_module(self, inputs, squeeze_depth, expand_depth, name):
        with tf.variable_scope(name):
            squeezed_inputs = self.custom_conv2d(
                inputs,
                squeeze_depth,
                [1, 1],
                'squeeze')
            expand1x1 = self.custom_conv2d(
                squeezed_inputs,
                expand_depth,
                [1, 1],
                'expand1x1')
            expand3x3 = self.custom_conv2d(
                squeezed_inputs,
                expand_depth,
                [3, 3],
                'expand3x3')
            return tf.concat([expand1x1, expand3x3], axis=-1)
            
    # Stacking fire modules
    def multi_fire_module(self, layer, param_list):
        for params in param_list:
            layer = self.fire_module(layer, params[0], params[1], params[2])
        return layer
    

    # Max pooling layer wrapper
    def castom_max_pooling2d(self, inputs, name):
        return tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=[2, 2],
            strides=2, 
            name=name)

    # Model layers
    # inputs: [batch_size, resize_dim, resize_dim, 3]
    def model_layers(self, inputs, is_training):
        conv1 = self.custom_conv2d(
            inputs, 
            filters=64, 
            kernel_size=[3, 3], 
            name='conv1')
        pool1 = self.castom_max_pooling2d(
            conv1, 
            name='pool1')
        fire_params1 = [(32, 64, 'fire1'), 
                        (32, 64, 'fire2')]
        multi_fire1 = self.multi_fire_module(
            pool1, 
            fire_params1)
        pool2 = self.castom_max_pooling2d(
            multi_fire1, 
            name='pool2')
        fire_params2 = [
            (32, 128, 'fire3'),
            (32, 128, 'fire4')
        ]
        multi_fire2 = self.multi_fire_module(
            pool2,
            fire_params2)
        dropout1 = tf.layers.dropout(multi_fire2, rate=0.5,
            training=is_training)
   