import tensorflow as tf
import pathlib
import os

from multiprocessing import Pool

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from_scratch = False

dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
dataset_file = tf.keras.utils.get_file(origin=dataset_url, extract=True if from_scratch else False)
data_path = pathlib.Path(os.path.join(os.path.dirname(dataset_file), 'PetImages'))

for file in data_path.glob('*/*.db'):
    os.remove(file)
    print(f'Remove {str(file)}')

def check_image(image_path):
    try:
        img_bytes = tf.io.read_file(str(image_path))
        tf.io.decode_image(img_bytes)
    except tf.errors.InvalidArgumentError as e:
        print(f'Remove {str(image_path)}')
        os.remove(image_path)

if __name__ == '__main__':
    with Pool() as p:
        p.map(check_image, data_path.glob('*/*.jpg'))
