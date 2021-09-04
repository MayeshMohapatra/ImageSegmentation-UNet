from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    Conv2DTranspose,
    Concatenate,
    Input,
)
from tensorflow.keras.models import Model

# implementing on unet architecture


##Simple Conv block with 2 3x3 conv layers


def Convolution_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


# Encoder block for downsampling
"""Contracting path consists of repeated application of 3x3 conv(unpadded),
    each followed by ReLU and a 2x2 max pooling operation with stride 2 for downsampling 
    at each downsampling the number of feature channels is doubled."""
# x will be used as skip connection, used in decoder
def encoder_block(input, num_filters):
    x = Convolution_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


# Decoder block for upsampling,
"""Expansive path consisting of an upsampling of the feature map followed by a 
    a 2x2 conv(up-conv) that halves the number of feature channels. concatenation
    with the cropped feature map from the contracting path and two 3x3 conv layers"""
# skip_features comes from x from the encoder
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = Convolution_block(x, num_filters)
    return x


## Constructing the unet architecture,according to the arxiv paper
def build_unet(input_shape):
    inputs = Input(input_shape)
    ## Downsampling, left side of unet
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    ## Bridge connection layers
    b1 = Convolution_block(p4, 1024)
    ## Upsampling, right side of unet
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_unet(input_shape)
    model.summary()
