from keras.layers import Dense,Conv2D,Embedding,LSTM,concatenate,Input,Reshape
import keras.layers as layers
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras.optimizers import SGD
import keras
import numpy


optimize=SGD(lr=0.0000001)
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

conv_layer5 = layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same')(concat_2)
conv_layer5 = layers.BatchNormalization()(conv_layer4)
conv_layer5 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer4)

flat_layer = layers.Flatten()(conv_layer5)
output_layer = layers.Dense(4,activation='relu')(flat_layer)

Box_model=Model(inputs=input_image, outputs=output_layer)
Box_model.compile(loss='mse', optimizer=optimize, metrics=['accuracy'])

Box_model.load_weights('my_model_weights.h5')

fhandle = open('test.csv', 'r')
fw = open('result.csv', 'w')
i=0
for line in fhandle:
    i=i+1
    toks = line.split(',')
    img_file = toks[0]
    img_file = '/tmp/flipkart/test_images/' + img_file
    img = load_img(img_file, target_size=(160, 120))
    img = img_to_array(img)
    img = img / 255
    img = numpy.array([img])
    cords = Box_model.predict(img)
    cords = cords[0]
    if i%100 == 0:
        print(i)
    fw.write(toks[0]+','+str(int(cords[0]))+','+str(int(cords[1]))+','+str(int(cords[2]))+','+str(int(cords[3]))+'\n')

fw.close()
fhandle.close()
