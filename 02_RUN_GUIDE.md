# How to Run the Code
## Complete Setup and Execution Guide

---

## Project Location

Replace `[PROJECT_PATH]` with your actual project folder path:

| Platform | Example Path |
|----------|--------------|
| **macOS** | `/Users/yourname/MLDL project` or `~/MLDL project` |
| **Windows** | `C:\Users\YourName\MLDL project` or `%USERPROFILE%\MLDL project` |
| **Linux** | `/home/yourname/MLDL project` or `~/MLDL project` |

---

## Quick Start

```bash
# Navigate to project (use your actual path)
cd [PROJECT_PATH]

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the web dashboard
python app.py
```

---

## Method 1: Terminal / Command Line

### macOS / Linux

**Step 1: Open Terminal**

- macOS: Press `Cmd + Space`, type "Terminal", press Enter
- Linux: Press `Ctrl + Alt + T`

**Step 2: Navigate to Project**

```bash
cd [PROJECT_PATH]
```

**Step 3: Create Virtual Environment (First Time)**

```bash
python3 -m venv venv
```

**Step 4: Activate Environment**

```bash
source venv/bin/activate
```

You should see `(venv)` at the start of your command prompt.

**Step 5: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 6: Run Scripts**

```bash
python app.py
```

```bash
python tune_model.py
```

**Step 7: Deactivate When Done**

```bash
deactivate
```

---

### Windows (Command Prompt / PowerShell)

**Step 1: Open Command Prompt**

- Press `Win + R`, type `cmd`, press Enter
- Or search for "Command Prompt" in Start menu

**Step 2: Navigate to Project**

```cmd
cd [PROJECT_PATH]
```

**Step 3: Create Virtual Environment (First Time)**

```cmd
py -3.13 -m venv venv
```

**Step 4: Activate Environment**

```cmd
venv\Scripts\activate
```

**Step 5: Install Dependencies**

```cmd
pip install -r requirements.txt
```

**Step 6: Run Scripts**

```cmd
python app.py
```

```cmd
python tune_model.py
```

**Step 7: Deactivate When Done**

```cmd
deactivate
```

---

## Method 2: VSCode

### Step 1: Open Project

```bash
code [PROJECT_PATH]
```

Or:
1. Open VSCode
2. File → Open Folder
3. Select your project folder

### Step 2: Select Python Interpreter

1. Press `Cmd + Shift + P` (macOS) or `Ctrl + Shift + P` (Windows/Linux)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `venv` (should show `Python 3.x.x ('venv': venv)`)

### Step 3: Run Script

**Option A: Run Button**
1. Open `fraud_detection.py`
2. Click the "Run" button in the top-right (▶️)

**Option B: Integrated Terminal**
1. View → Terminal (or press `` Ctrl + ` ``)
2. Run commands as shown in Terminal method

---

## Method 3: Jupyter Notebook

### Step 1: Install Jupyter

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install jupyter
```

### Step 2: Launch Notebook

```bash
jupyter notebook
```

Opens in browser at `http://localhost:8888`

### Step 3: Create Notebook

1. Click "New" → "Python 3"
2. Import functions:

```python
import sys
sys.path.append('[PROJECT_PATH]')

from fraud_detection import preprocess_for_nn, build_model, train_model, evaluate_model
```

---

## Running Different Scripts

### app.py (Web Dashboard - Main Entry Point)

```bash
python app.py
```

**What it does:**
1. Serves the interactive HTML frontend using Flask.
2. Accepts manual inputs or runs data dynamically against the neural network.
3. Automatically loads `fraud_model.keras` and `preprocessor.pkl`.
4. Renders live architecture visualization, training curves, and EDA plots.

**Open your browser to: http://127.0.0.1:5000**

---

### fraud_detection.py (Main Training)

```bash
python fraud_detection.py
```

**What it does:**
1. Loads training/test data
2. Preprocesses features (engineering, encoding, scaling)
3. Trains neural network (256→128→64)
4. Evaluates on test set
5. Saves model and preprocessor

**Output files:**
- `fraud_model.keras` - Trained model
- `preprocessor.pkl` - Encoders and scaler

---

### tune_model.py (Hyperparameter Tuning)

```bash
python tune_model.py
```

**What it does:**
1. Runs 15 random search trials
2. Tests different architectures, dropout rates, learning rates
3. Finds best hyperparameters

**Output files:**
- `hyperparam_results.csv` - All trial results
- `best_hyperparams.pkl` - Best parameters

**Expected time:** 10-30 minutes

---

## Troubleshooting

### "python: command not found"

Use `python3` instead:

```bash
python3 fraud_detection.py
```

---

### "No module named 'tensorflow'"

Ensure virtual environment is activated:

```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

---

### Out of Memory

Use smaller data samples for testing:

```python
# In fraud_detection.py main(), change:
df_train = pd.read_csv('fraudTrain.csv', nrows=100000)
df_test = pd.read_csv('fraudTest.csv', nrows=10000)
```

---

### Slow Training

**Reduce batch size:**
```python
batch_size=1024
```

**Reduce epochs:**
```python
epochs=10
```

**Use smaller model:**
```python
hidden_layers=[128, 64]
```

---

## Environment Setup Summary

| Step | macOS/Linux | Windows |
|------|-------------|---------|
| Create venv | `python3 -m venv venv` | `python -m venv venv` |
| Activate | `source venv/bin/activate` | `venv\Scripts\activate` |
| Install | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Run | `python app.py` | `python app.py` |
| Deactivate | `deactivate` | `deactivate` |

---

## File Structure

```
├── app.py                       # Flask Web Dashboard Main Entry Point
├── templates/                   # HTML Frontend code
├── static/                      # CSS Styling 
├── fraud_model.keras            # Trained model
├── preprocessor.pkl             # Encoders and scaler
├── fraud_detection.py           # Model training implementation
├── tune_model.py                # Hyperparameter tuning
├── requirements.txt             # Dependencies
├── 01_THEORY_TUTORIAL.md        # Complete theory guide
├── 03_RUN_GUIDE.md              # How to run (this file)
├── 04_CODE_GUIDE_fraud_detection.md  # fraud_detection.py explained
├── 05_CODE_GUIDE_tune_model.md   # tune_model.py explained
├── fraudTrain.csv               # Training data (1.3M rows)
└── fraudTest.csv                # Test data (555K rows)
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Activate (macOS/Linux) | `source venv/bin/activate` |
| Activate (Windows) | `venv\Scripts\activate` |
| Install deps | `pip install -r requirements.txt` |
| Run dashboard | `python app.py` |
| Run main training | `python fraud_detection.py` |
| Run tuning | `python tune_model.py` |
| Deactivate | `deactivate` |