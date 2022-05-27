import tensorflow as tf
from tensorflow import keras
from tensorflow import layers

# input:
# map flatten 64 * 64 of [0,1,2,3]
# 0 - free
# 1 - obst
# 2 - foreign agent
# 3 - myself
# deikstras output
# target vector
# target distance

N = 256

loss_tracker = keras.metrics.Mean(name='loss')
mae_metric = keras.metrics.MeanAbsoluteError(name='mae')

class CustomModel(keras.Model):
    self.model = None
    def make_model(self, input_map_shape,input_misc_shape,out_states):
        inputs1 = keras.Input(shape=input_map_shape)
        inputs2 = keras.Input(shape=input_misc_shape)
        x1 = inputs1
        print(x1.shape)
        x1 = layers.Reshape((64,64,-1))(x)
        print(x1.shape)
        x1 = layers.Conv2D(3,3,activation='relu',padding='same')(x1)
        print(x1.shape)
        x1 = layers.Conv2D(3,3,activation='relu',padding='same')(x1)
        print(x1.shape)
        x2 = inputs2
        print(x2.shape)
        x2 = layers.Dense(20,activation='sigmoid')(x2)
        print(x2.shape)
        x2 = layers.Dense(10,activation='sigmoid')(x2)
        x = keras.Concatenate(axis=1)([x1,x2])
        x = layers.Dense(600,activation='sigmoid')(x)
        print(x.shape)
        x = layers.Dense(100,activation='sigmoid')(x)
        print(x.shape)
        outputs = layers.Dense(5,activation='sigmpoid')(x)
        self.model = keras.Model(inputs,outputs)

    def make_prediction(self,mp):
        out = self.model.predict(mp)
        return out

    def train(self, epochs):
        callbacks = {
            keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")
        }
        self.model.compile(
            keras.optimizers.Adam(1e-3),
            loss='mse', # mse
            metrics=['mae'])
        self.model.fit(
            # [map,misc] = x
            gen_data(),None,
            epochs=epochs
        )
    def function():
        pass
    def get_score():
        pass
    def get_max_score(data):
        pass
    def train_step(self, inp_data):
        data,y = inp_data
        with tf.GradientTape() as tape:
            for i in range(N):
                moves = act(*data)
                update(data,moves)
        result = get_score(data)
        loss = keras.losses.mean_squared_error(len(data),result) # or binary_crossentropy
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss,trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)
        mae_metric.update_state(len(data),result)
        return {'loss': loss_tracker.result(), 'mae': mae_metric.result()}
