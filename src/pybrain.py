import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_data = np.array([-40.0, 14.0, 32.0, 46.0])
output_data = np.array([233.15, 263.15, 273.15, 280.928])

layers = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layers])

model.compile(loss=tf.losses.MeanSquaredError, optimizer=tf.keras.optimizers.Adam(.1))
history = model.fit(input_data, output_data, epochs=9000)

for i, element in enumerate(input_data):
    print(f'F {element} - K {output_data[i]} - M {model.predict([element])}')
