import tensorflow as tf
import numpy as np

class FlowerPipeline:
    def __init__(self, img_size=(180, 180), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = None

    def _parse_function(self, filename, label):
        """
        Data Engineering: Reading, Decoding, and Scaling.
        """
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, self.img_size)

        image = image / 255.0 
        return image, label

    def create_dataset(self, df):
        """
        Converts Pandas DataFrame to a high-performance tf.data Dataset.
        """
        if self.class_names is None:
            self.class_names = sorted(df['label'].unique())

        labels = df['label'].map({name: i for i, name in enumerate(self.class_names)}).values
        filenames = df['image_path'].values

        ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return ds

if __name__ == "__main__":
    print("Pipeline module loaded. Ready to transform Series data into Tensors.")
