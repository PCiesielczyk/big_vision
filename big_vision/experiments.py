import tensorflow as tf
import matplotlib.pyplot as plt

tfrecord_path = "C:\\Users\piotr\\tensorflow_datasets\\cifar10\\3.0.2\\cifar10-train.tfrecord-00000-of-00001"

# Funkcja do parsowania rekordów
def parse_tfrecord_fn(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),  # Obraz zakodowany jako string (JPEG/PNG)
        "label": tf.io.FixedLenFeature([], tf.int64),  # Etykieta jako int64
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Dekodowanie obrazu (zakładamy format JPEG; zmień na decode_png w razie potrzeby)
    image = tf.image.decode_jpeg(parsed_example["image"], channels=3)  # Rozpakowanie JPEG do (wysokość, szerokość, 3)
    label = parsed_example["label"]

    # Dopasowanie rozmiaru obrazu do 32x32 (w razie potrzeby)
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.uint8)  # Konwersja na uint8 dla wyświetlania

    return image, label


# Ładowanie i przetwarzanie danych
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

# Wyświetlenie kilku obrazków
plt.figure(figsize=(10, 5))
for i, (image, label) in enumerate(parsed_dataset.take(5)):  # Wyświetl pierwsze 5 obrazków
    plt.subplot(1, 5, i + 1)
    plt.imshow(tf.squeeze(image).numpy().astype("uint8"))
    plt.title(f"Label: {label.numpy()}")
    plt.axis("off")
plt.tight_layout()
plt.show()
