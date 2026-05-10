import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

class DataEngine:
    def __init__(self, dataset_url):
        self.dataset_url = dataset_url
        self.data_dir = None

    def download_data(self):
        """Downloads and extracts the dataset."""
        self.data_dir = tf.keras.utils.get_file('flower_photos', origin=self.dataset_url, untar=True)
        return self.data_dir

    def create_metadata_df(self):
        # Path where Keras downloads the data
        base_path = self.data_dir 
        
        # Check if there is a nested 'flower_photos' folder
        nested_path = os.path.join(base_path, 'flower_photos')
        if os.path.exists(nested_path):
            base_path = nested_path
        
        # Get actual flower directories (Daisy, Rose, etc.)
        flower_labels = [f for f in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, f))]
        
        data_list = []
        for label in flower_labels:
            path = os.path.join(base_path, label)
            for img in os.listdir(path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    data_list.append({
                        'image_path': os.path.join(path, img),
                        'label': label
                    })
        
        df = pd.DataFrame(data_list)
        
        if df.empty:
            raise ValueError(f"No images found in {base_path}. Current directories: {os.listdir(base_path)}")
            
        return df



    def clean_and_split(self, df, test_size=0.2):
        """
        Applies cleaning and prevents Data Leakage by splitting 
        before any synthetic feature generation.
        """
        df = df[df['image_path'].str.endswith(('.jpg', '.jpeg', '.png'))]

        train_df, val_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['label'], 
            random_state=42
        )
        return train_df, val_df

if __name__ == "__main__":
    url = 'https://googleapis.com'
    engine = DataEngine(url)
    path = engine.download_data()
    raw_df = engine.create_metadata_df()
    train, val = engine.clean_and_split(raw_df)
    print(f"Engine Ready. Train size: {len(train)}, Val size: {len(val)}")
