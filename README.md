# Controllable Syllable-Level Lyrics Generation From Melody With Prior Attention

This repository contains the code and resources for the paper [**"Controllable Syllable-Level Lyrics Generation From Melody With Prior Attention"** (IEEE Trans. Multimedia, 2024)](https://ieeexplore.ieee.org/document/10637751). The project proposes:

* A syllable-level melody-to-lyrics Transformer model with a **melody encoder**, **lyrics decoder**, **explicit N-Gram (EXPLING) loss**, and **prior attention** mechanism.
* Controllable and diverse lyrics generation guided by user-specified rhythm patterns.
* Preprocessing, training, inference, and evaluation pipelines for reproducible research.

---

## Repository Structure

```
├── data/                          # Raw and processed datasets
├── tokenizers/                    # Saved tokenizer objects
├── preprocess.py                  # Data preprocessing script
├── dataset_features_flags.py      # Dataset and DataModule definitions
├── n_gram_loss.py                 # ExplicitNGramCriterion (EXPLING loss)
├── models.py                      # Transformer model implementations
├── train.py                       # Training script with PyTorch Lightning
├── inference.py                   # Inference script for lyrics and MIDI output
├── evaluate.py                    # Evaluation script for metrics
├── hparams.py                     # Hyperparameters configuration
└── utils.py                       # Utility functions (tokenizer I/O, MIDI creation)
```

---

## Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/lyrics-generation.git
   cd lyrics-generation
   ```
2. Create a Conda environment (Python 3.8+):

   ```bash
   conda create -n con-prattn python=3.8
   conda activate con-prattn
   pip install -r requirements.txt
   ```
3. Install additional dependencies if reported.
---

## Data Preprocessing

Run the preprocessing script to build fixed-length sequences, flags for prior attention, and fit tokenizers:

```bash
python preprocess.py 
```

Outputs:

* `sequences.npy`, train/valid/test splits
* Tokenizers saved in `tokenizers/`

Note: The raw melody-lyrics dataset is not included in this repository.
Please obtain the data separately and run the preprocess.py pipeline to generate the required npy files.
Please check the details in the codes and adapt them to your dataset.

---

## Training

Train the Transformer model with EXPLING loss and prior attention:

```bash
python train.py 
```

Checkpoints saved under `./checkpoints/` and logged via Neptune.

Please check the arguments in the codes.

---

## Inference

Generate syllable-level lyrics and a MIDI file:

```bash
python inference.py 
```

Override prior attention patterns in `inference.py` to control rhythm.

---

## Evaluation

Compute automatic metrics on the test set:

```bash
python evaluate.py 
```

Metrics:

* ROUGE-1/2/L (precision, recall, F1)
* Sentence BLEU-2/3/4 and Corpus BLEU
* ChrF and ChrF++
* BERTScore (P/R/F1)
* InfoLM (KL divergence & L2 distance)

Results printed and saved under `./eval_results/`.

---

## Citation

If you use this code, please cite:

```bibtex
@ARTICLE{Zhang2024Controllable,
  author={Z. Zhang and Y. Yu and A. Takasu},
  journal={IEEE Transactions on Multimedia},
  title={Controllable Syllable-Level Lyrics Generation From Melody With Prior Attention},
  year={2024},
  volume={26},
  pages={11083-11094},
  doi={10.1109/TMM.2024.3443664}
}
```
