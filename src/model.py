from tensorflow.keras import layers, models

class FlowerCNN:
    def __init__(self, input_shape=(180, 180, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        """
        Constructs a CNN architecture using Conv2D, BatchNormalization, 
        ReLU, and Dropout layers.
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),

            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

if __name__ == "__main__":
    flower_model = FlowerCNN().build_model()
    flower_model.summary()
