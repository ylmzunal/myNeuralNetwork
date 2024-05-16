import numpy as np
from neural_Network import NeuralNetwork, Layer, ActivationFunction
from keras.datasets import mnist
from keras.utils import to_categorical

# MNIST veri setini yükleyin
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Verileri normalize edin (0-255 arasından 0-1 arasına)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Giriş verilerini düzleştirin (28x28 -> 784)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Etiketleri one-hot encode edin
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Sinir ağı modelini oluşturun
nn = NeuralNetwork()
nn.add_layer(Layer(input_size=784, output_size=128, activation_function=ActivationFunction.sigmoid))
nn.add_layer(Layer(input_size=128, output_size=10, activation_function=ActivationFunction.sigmoid))

# Modeli eğitin
nn.train(x_train, y_train, epochs=10000, learning_rate=0.1)

# Modeli test edin
predictions = nn.predict(x_test)
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
print(f'Test Accuracy: {accuracy * 100:.2f}%')

