import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import librosa

SAMPLE_RATE = 22050
N_MELS = 128

def preprocess_audio(file_path, sr=SAMPLE_RATE, n_mels=N_MELS, duration=1.0):
    """Converts audio to log mel spectrogram."""
    # Load the audio file
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    
    # Zero-pad/trim the audio signal to the same length
    if len(y) < sr * duration:
        y = np.pad(y, (0, int(sr * duration) - len(y)))
    else:
        y = y[:int(sr * duration)]
    
    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spectrogram

def process_directory(directory_path):
    """Processes all files in a directory and returns a list of their log mel spectrograms."""
    spectrograms = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            spectrogram = preprocess_audio(file_path)
            spectrograms.append(spectrogram)
    return spectrograms

def create_dataset(kicks, snares, hi_hats):
    X = np.concatenate([kicks, snares, hi_hats])
    y = np.concatenate([
        np.zeros(len(kicks)),
        np.ones(len(snares)),
        2*np.ones(len(hi_hats))
    ])
    return X, y


def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # Three classes: Kick, Snare, Hi-hat
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    samples_path = os.path.join(base_path, '../samples')

    kicks_spectrograms = process_directory(os.path.join(samples_path, 'kicks'))
    snares_spectrograms = process_directory(os.path.join(samples_path, 'snares'))
    hi_hats_spectrograms = process_directory(os.path.join(samples_path, 'hats'))

    # Create dataset
    X, y = create_dataset(kicks_spectrograms, snares_spectrograms, hi_hats_spectrograms)
    X = X[..., np.newaxis]  # Adding a channel dimension for CNN

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Define model
    model = build_model(X_train[0].shape)

    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("\nTest accuracy:", test_acc)
