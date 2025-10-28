import librosa  # type: ignore
import numpy as np
import pandas as pd # type: ignore
import os
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm # type: ignore


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)  

        #MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        #Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        #Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        #Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid, axis=1)

        #Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Combine all features
        features = np.hstack([
            mfcc_mean,
            chroma_mean,
            contrast_mean,
            centroid_mean,
            tempo
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}:{e}")
        return np.zeros(34)  # placeholder for failed files


if __name__ == "__main__":
    path = r"D:\recomdn_engine\Data\genres_original\*\*.wav"
    files = glob(path)

    # Multiprocessing setup
    num_cores = cpu_count() - 2  # keep 2 cores free
    print(f"Using {num_cores} CPU cores for parallel processing...")

    # Run multiprocessing with progress bar
    with Pool(num_cores) as p:
        results = list(tqdm(p.imap(extract_features, files), total=len(files)))

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df['filename'] = [os.path.basename(f) for f in files]
    df['genre'] = [os.path.basename(os.path.dirname(f)) for f in files]

    df.drop(index=554,inplace=True)

    # Save to CSV
    df.to_csv("audio_features.csv", index=False)
    print("Feature extraction complete! Saved to 'audio_features.csv'")

