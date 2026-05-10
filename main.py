import wandb
from wandb.integration.keras import WandbMetricsLogger # New Callback
from src.data_engine import DataEngine
from src.pipeline import FlowerPipeline
from src.model import FlowerCNN
from tensorflow.keras.callbacks import EarlyStopping


URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 20

def train_model():
    wandb.init(project="flower-recognition", config={
        "learning_rate": 0.001,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "architecture": "CNN-Vanilla"
    })
    config = wandb.config

    engine = DataEngine(URL)
    engine.download_data()
    df = engine.create_metadata_df()
    train_df, val_df = engine.clean_and_split(df)

    pipeline = FlowerPipeline(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    train_ds = pipeline.create_dataset(train_df)
    val_ds = pipeline.create_dataset(val_df)

    num_classes = len(train_df['label'].unique())
    model_builder = FlowerCNN(input_shape=(*IMG_SIZE, 3), num_classes=num_classes)
    model = model_builder.build_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=[early_stop, WandbMetricsLogger()] # Updated here
    )


    model.save("flower_model.keras")
    print("Model trained and saved as flower_model.keras")

if __name__ == "__main__":
    train_model()
