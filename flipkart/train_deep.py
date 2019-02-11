from keras.layers import Dense,Conv2D,Embedding,LSTM,concatenate,Input,Reshape
import keras.layers as layers
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras.optimizers import SGD,Adam
import keras
import numpy

def data_generate():
    image = []
    co_ordinates = []
    while 1:
        fhandle = open('training.csv','r')
        num_images=0
        for line in fhandle:
            num_images = num_images + 1
            line_tok = line.split(',')
            img_name = '/tmp/flipkart/training_images/'+line_tok[0]
            img = load_img(img_name,target_size=(160,120))
            img = img_to_array(img)
            img = img/255
            points = [int(line_tok[1]),int(line_tok[2]),int(line_tok[3]),int(line_tok[4])]
            image.append(img)
            co_ordinates.append(points)
            if num_images%20 == 0:
                image = numpy.array(image)
                co_ordinates = numpy.array(co_ordinates)
                yield [image, co_ordinates]
                image = []
                co_ordinates = []
        fhandle.close()

optimize=Adam(lr=0.000001)
input_image = Input(shape=(160,120,3))

conv_layer1 = layers.Conv2D(32, (3, 3), strides=(1, 1),padding='same')(input_image)
conv_layer1 = layers.BatchNormalization()(conv_layer1)
conv_layer1 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer1)

conv_layer2 = layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same')(conv_layer1)
conv_layer2 = layers.BatchNormalization()(conv_layer2)
conv_layer2 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer2)

concat_1 = layers.concatenate([conv_layer1,conv_layer2])

conv_layer3 = layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same')(concat_1)
conv_layer3 = layers.BatchNormalization()(conv_layer3)
conv_layer3 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer3)

conv_layer4 = layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same')(conv_layer3)
conv_layer4 = layers.BatchNormalization()(conv_layer4)
conv_layer4 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer4)

concat_2 = layers.concatenate([conv_layer3,conv_layer4])

conv_layer5 = layers.Conv2D(32, (3, 3), strides=(1, 1),padding='same')(concat_2)
conv_layer5 = layers.BatchNormalization()(conv_layer5)
conv_layer5 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer5)

conv_layer6 = layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same')(conv_layer5)
conv_layer6 = layers.BatchNormalization()(conv_layer6)
conv_layer6 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer6)

concat_3 = layers.concatenate([conv_layer5,conv_layer6])

conv_layer7 = layers.Conv2D(32, (3, 3), strides=(1, 1),padding='same')(concat_3)
conv_layer7 = layers.BatchNormalization()(conv_layer7)
conv_layer7 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer7)

conv_layer8 = layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same')(conv_layer7)
conv_layer8 = layers.BatchNormalization()(conv_layer8)
conv_layer8 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer8)

concat_4 = layers.concatenate([conv_layer7,conv_layer8])

conv_layer9 = layers.Conv2D(32, (3, 3), strides=(1, 1),padding='same')(concat_4)
conv_layer9 = layers.BatchNormalization()(conv_layer9)
conv_layer9 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer9)

conv_layer10 = layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same')(conv_layer9)
conv_layer10 = layers.BatchNormalization()(conv_layer10)
conv_layer10 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer10)

concat_5 = layers.concatenate([conv_layer9,conv_layer10])

conv_layer11 = layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same')(concat_5)
conv_layer11 = layers.BatchNormalization()(conv_layer11)
conv_layer11 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer11)

flat_layer = layers.Flatten()(conv_layer11)
output_layer = layers.Dense(4,activation='relu')(flat_layer)

Box_model=Model(inputs=input_image, outputs=output_layer)
Box_model.compile(loss='mse', optimizer=optimize, metrics=['accuracy'])

Box_model.load_weights('my_model_weights_v2.h5')
train_generator=data_generate()
Box_model.fit_generator(generator=train_generator,steps_per_epoch=700,epochs=5)
Box_model.save_weights('my_model_weights_v2.h5')
