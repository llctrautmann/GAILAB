# Audio Phase Reconstruction

This repository contains the implementation of a project that explores the use of self-supervised machine learning architectures to approximate the phase component of a Discrete Fourier Transform (DFT) for rendering high-resolution reconstructions of ecosystem audio recordings. The goal of this project is to enable the audible interpretation of network outputs generated through reconstruction learning in the field of ecoacoustics.

## Background

Monitoring and understanding the integrity of our planetary biosphere is crucial for addressing sustainability issues. The emerging field of Ecoacoustics offers a promising approach by allowing us to eavesdrop on ecosystems and assess their health. However, in order to fully interpret and analyze the audio data collected from ecosystems, it is essential to have reconstructions that include both the magnitude and phase components of the audio signals.

Signal reconstruction from magnitude-only representations has been a longstanding problem in signal processing. Traditional iterative algorithms for phase estimation often produce noisy or artifact-ridden results, which hinders the reliable interpretation of audible features in an ecoacoustic context. This project aims to address this challenge by leveraging self-supervised machine learning architectures to approximate the missing phase component and generate high-resolution reconstructions of ecosystem audio recordings.

## Project Objectives

The main objectives of this project are as follows:

1. Develop a self-supervised machine learning architecture for approximating the phase component of a DFT.
2. Train the model using audio data from ecosystem recordings, with a focus on ecoacoustic applications.
3. Generate high-resolution reconstructions of the audio signals by combining the approximated phase component with the magnitude component.
4. Evaluate the quality and fidelity of the reconstructions through quantitative and qualitative measures.
5. Provide ecologists and researchers with a tool for better understanding and interpreting audible features in ecoacoustic predictions.

## Repository Structure

The repository is organized as follows:

- `data/`: This directory contains the audio datasets used for training and evaluation. Due to the size limitations of GitHub, the actual audio files are not included in the repository.
- `src/`: This directory contains the implementation of the self-supervised machine learning architecture for phase approximation.
- `deprecated/`: This directory contains old versions of the code base, kept for legacy purposes
- `reconstruction/`: This directory contains the code for combining the magnitude and approximated phase components to generate high-resolution reconstructions.
- `evaluation/`: This directory contains scripts and notebooks for evaluating the quality and fidelity of the reconstructions.
- `docs/`: This directory contains any relevant documentation, including user guides or tutorials.
- `notebooks/`: This directory contains example code snippets or notebooks to demonstrate the usage of the implemented models and reconstruction techniques.
- `LICENSE`: The license file for this repository.
- `README.md`: The main README file that provides an overview of the project.

## Getting Started

To get started with this project, please follow the instructions below:

1. Clone the repository to your local machine.
2. Install the required dependencies and libraries as specified in the toml file.
3. Obtain the audio datasets for training and evaluation. Ensure that the dataset files are stored in the appropriate directory (`data/`).
4. Run the training script or notebook to train the self-supervised machine learning model on the audio datasets.
5. Use the trained model to approximate the phase component of the audio signals and generate reconstructions by running the provided reconstruction script or notebook.
6. Evaluate the quality and fidelity of the reconstructions using the evaluation scripts or notebooks available in the `evaluation/` directory.
7. Refer to the documentation in the `docs/` directory for additional information on using the code and interpreting the results.

## Contribution Guidelines

Contributions to this project are welcome. If you would like to contribute, please follow these guidelines:

1. Fork the repository and create a new branch for your contribution.
2. Make your changes or additions in the new branch.
3.

 Ensure that your code is well-documented and adheres to the project's coding style guidelines.
4. Test your changes thoroughly.
5. Submit a pull request, describing the nature of your contribution and any relevant information.



