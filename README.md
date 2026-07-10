# Audio Matching Using FFT

A Python project that locates a short audio clip inside a longer WAV recording using the Fast Fourier Transform and a sliding-window similarity algorithm.

## Project Idea

The program extracts a query clip from a full audio signal and searches for its position inside the recording.

Each window of the full signal is converted from the time domain to the frequency domain using FFT. The frequency spectrum of each window is compared with the query clip using a normalized dot product.

The position with the highest similarity score is selected as the detected clip position.

## Features

- Loads WAV audio files
- Converts stereo audio to mono
- Normalizes the audio signal
- Plots signals in the time domain
- Computes one-sided FFT magnitude
- Plots signals in the frequency domain
- Uses a sliding-window matching algorithm
- Calculates normalized similarity scores
- Detects the clip position
- Compares the original clip with the detected segment

## Technologies Used

- Python
- NumPy
- SciPy
- Matplotlib
- Spyder

## Project Configuration

- Full audio duration: approximately 60 seconds
- Query clip position: 25 seconds
- Query clip duration: 5 seconds
- Sliding-window step size: 0.1 seconds

## Program Output

The program prints:

- Sampling frequency
- Full signal length
- Clip length
- Original clip position
- Detected position
- Best similarity score

## How to Run

1. Install the required libraries:

pip install numpy scipy matplotlib
2. Place a WAV file named audio.wav in the same folder as the Python file.
3. Run: python audio_matching_fft.py

Course

Signals and Systems Theory — COMM 401

Authors

* Hamza Wael Sallam
* Omar Moataz El Sobky
