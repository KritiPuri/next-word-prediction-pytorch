# Next Word Prediction Using LSTM

## What's included
- `hamlet.txt` — training text
- `tokenizer.py` — simple word tokenizer and vocabulary
- `model.py` — PyTorch embedding + 2 LSTM layers + dropout + output layer
- `train.py` — training pipeline, validation split, early stopping, checkpoint save
- `predict.py` — model loading and next-word prediction
- `app.py` — Streamlit UI
- `requirements.txt` — PyTorch, NumPy, Streamlit only

No TensorFlow, Keras, `.h5`, or TensorFlow tokenizer/pickle artifacts are used.

## 1. Setup
Create and activate a virtual environment:

### Windows
```bash
py -m venv venv
venv\\Scripts\\activate
```

### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Train the model
Run from the project directory:
```bash
python train.py
```
Optional:
```bash
python train.py --epochs 50 --batch-size 128 --patience 3
```
Training automatically uses CUDA if available; otherwise CPU. The best validation-loss model is saved to `artifacts/next_word_lstm.pt`, along with `artifacts/tokenizer.json`.

## 3. Predict in terminal
After training:
```bash
python predict.py
```

## 4. Launch the web app
```bash
streamlit run app.py
```

## Architecture
- Embedding: 100 dimensions
- LSTM layer 1: 150 hidden units, returns sequence
- Dropout: 0.2
- LSTM layer 2: 100 hidden units
- Linear output: vocabulary size
- Loss: CrossEntropyLoss (equivalent to categorical cross-entropy with integer labels)
- Optimizer: Adam
- Early stopping: monitors validation loss, patience 3, restores best weights

