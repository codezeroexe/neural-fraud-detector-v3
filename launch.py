#!/usr/bin/env python3
"""
Neural Fraud Detector v2 - GUI Launcher

Cross-platform GUI app that:
- Downloads dataset automatically if not present
- Launches dashboard in browser
- No manual download needed
"""

import os
import sys
import subprocess
import webbrowser
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox
    from tkinter import ttk
    from tkinter import scrolledtext
except ImportError:
    tk = None

# Configuration
APP_PORT = 5000
APP_URL = f"http://127.0.0.1:{APP_PORT}"
VENV_PATH = "venv"

TRAIN_FILE = "fraudTrain.csv"
TEST_FILE = "fraudTest.csv"
MODEL_FILE = "fraud_model.keras"

# Alternative download URLs (direct links - may need update)
DATA_URLS = [
    "https://github.com/codezeroexe/fraud-data/raw/main/fraudTrain.csv",
    "https://github.com/codezeroexe/fraud-data/raw/main/fraudTest.csv",
]


def get_project_dir():
    return Path(__file__).parent.resolve()


def check_files():
    project_dir = get_project_dir()
    return (project_dir / TRAIN_FILE).exists() and (project_dir / MODEL_FILE).exists()


def check_data():
    project_dir = get_project_dir()
    return (project_dir / TRAIN_FILE).exists()


def check_model():
    project_dir = get_project_dir()
    return (project_dir / MODEL_FILE).exists()


def download_file(url, dest_path, progress_callback=None):
    """Download a file with progress."""
    try:
        urllib.request.urlretrieve(url, dest_path, progress_callback)
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False


def open_browser():
    time.sleep(1)
    webbrowser.open(APP_URL)


def find_python():
    project_dir = get_project_dir()
    venv_python = project_dir / VENV_PATH / "bin" / "python"
    if sys.platform == "win32":
        venv_python = project_dir / VENV_PATH / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def find_pip():
    project_dir = get_project_dir()
    venv_pip = project_dir / VENV_PATH / "bin" / "pip"
    if sys.platform == "win32":
        venv_pip = project_dir / VENV_PATH / "Scripts" / "pip.exe"
    if venv_pip.exists():
        return str(venv_pip)
    return "pip"


def setup_environment(root=None, status_var=None, progress_var=None):
    """Set up venv and install deps."""
    project_dir = get_project_dir()
    python = find_python()
    pip = find_pip()
    
    # Create venv
    if not (project_dir / VENV_PATH).exists():
        update_status(root, status_var, "Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], 
                    cwd=project_dir, capture_output=True)
    
    # Install deps
    update_status(root, status_var, "Installing dependencies...")
    subprocess.run([pip, "install", "-r", "requirements.txt", "--quiet"],
                 cwd=project_dir, capture_output=True)


def update_status(root, var, text):
    if var and root:
        var.set(text)
        root.update()


def download_dataset(root, status_var, progress_var, progress_bar):
    """Auto-download dataset."""
    project_dir = get_project_dir()
    
    # Try Kaggle API first
    try:
        update_status(root, status_var, "Checking for Kaggle...")
        import kaggle
        update_status(root, status_var, "Downloading from Kaggle...")
        kaggle.api.dataset_download_files('kartik2112/fraud-detection', 
                                   path=str(project_dir), unzip=True)
        return True
    except:
        pass
    
    # Fallback: Try direct download URLs
    for url in DATA_URLS:
        update_status(root, status_var, f"Downloading... {url}")
        filename = url.split('/')[-1]
        dest = project_dir / filename
        
        try:
            # Manual download with progress
            def report(block, block_size, total_size):
                if total_size > 0:
                    percent = min(100, int(block * block_size * 100 / total_size))
                    if progress_bar:
                        progress_bar['value'] = percent
                    if progress_var:
                        progress_var.set(f"{percent}%")
                    root.update()
            
            urllib.request.urlretrieve(url, dest, report)
            
            if (project_dir / filename).exists():
                update_status(root, status_var, f"Downloaded {filename}")
                return True
        except Exception as e:
            continue
    
    return False


def train_model(root, status_var):
    """Train model if not present."""
    if not check_model():
        update_status(root, status_var, "Training model (first time)...")
        project_dir = get_project_dir()
        python = find_python()
        subprocess.run([python, "fraud_detection.py"], 
                     cwd=project_dir, capture_output=True)


def run_flask():
    """Run Flask app."""
    project_dir = get_project_dir()
    python = find_python()
    subprocess.run([python, "app.py"], cwd=str(project_dir),
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def gui_launch():
    """Main GUI launcher."""
    if tk is None:
        terminal_fallback()
        return
    
    project_dir = get_project_dir()
    os.chdir(project_dir)
    
    # Create window
    root = tk.Tk()
    root.title("Neural Fraud Detector v2")
    root.geometry("420x380")
    root.resizable(False, False)
    root.eval('tk::PlaceWindow . center')
    
    # Variables
    status_var = tk.StringVar(value="Starting...")
    progress_var = tk.StringVar(value="")
    
    # UI
    icon_label = tk.Label(root, text="🛡️", font=("Arial", 52))
    icon_label.pack(pady=10)
    
    title_label = tk.Label(root, text="Neural Fraud Detector v2", font=("Arial", 16, "bold"))
    title_label.pack()
    
    status_label = tk.Label(root, textvariable=status_var, font=("Arial", 11), fg="#333")
    status_label.pack(pady=15)
    
    progress_label = tk.Label(root, textvariable=progress_var, font=("Arial", 10), fg="#666")
    progress_label.pack()
    
    progress_bar = ttk.Progressbar(root, mode='determinate', length=300)
    progress_bar.pack(pady=10)
    
    log_text = scrolledtext.ScrolledText(root, height=6, width=45, font=("Courier", 9))
    log_text.pack(pady=10)
    log_text.config(state='disabled')
    
    def log(msg):
        log_text.config(state='normal')
        log_text.insert(tk.END, f"{msg}\n")
        log_text.see(tk.END)
        log_text.config(state='disabled')
        root.update()
    
    root.update()
    
    # Check files
    if check_files():
        status_var.set("Ready! Launching...")
        log("Files found")
        
        # Setup if needed
        setup_environment(root, status_var)
        
        # Open browser
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Run
        threading.Thread(target=run_flask, daemon=True).start()
        
        log(f"Dashboard: {APP_URL}")
        messagebox.showinfo("Ready", f"Open: {APP_URL}")
        root.after(2000, root.destroy)
        root.mainloop()
        return
    
    # Need setup
    status_var.set("Setting up...")
    log("Starting setup...")
    
    try:
        # Setup environment
        setup_environment(root, status_var)
        log("Environment ready")
        
        # Download data
        if not check_data():
            status_var.set("Downloading dataset...")
            log("Downloading dataset...")
            
            # Try multiple methods
            success = False
            
            # Method 1: Try direct URLs
            for url in DATA_URLS:
                filename = url.split('/')[-1]
                dest = project_dir / filename
                try:
                    log(f"Trying: {filename}")
                    urllib.request.urlretrieve(url, dest)
                    if dest.exists():
                        success = True
                        log(f"Downloaded {filename}")
                        break
                except:
                    continue
            
            if not success:
                # Method 2: Manual download prompt
                log("Please download dataset manually:")
                log("https://www.kaggle.com/datasets/kartik2112/fraud-detection")
                result = messagebox.askyesno(
                    "Download Required",
                    "Auto-download failed.\n\n"
                    "1. Go to: kaggle.com/datasets/kartik2112/fraud-detection\n"
                    "2. Download and extract\n"
                    "3. Place fraudTrain.csv in project folder\n\n"
                    "Click Yes to open download page."
                )
                if result:
                    webbrowser.open("https://www.kaggle.com/datasets/kartik2112/fraud-detection")
                return
        else:
            log("Dataset found")
        
        # Train model
        train_model(root, status_var)
        log("Model ready")
        
        # Launch
        status_var.set("Launching dashboard...")
        threading.Thread(target=open_browser, daemon=True).start()
        threading.Thread(target=run_flask, daemon=True).start()
        
        log(f"Dashboard: {APP_URL}")
        messagebox.showinfo("Ready", f"Open: {APP_URL}")
        
    except Exception as e:
        log(f"Error: {e}")
        messagebox.showerror("Error", str(e))
    
    root.after(3000, root.destroy)
    root.mainloop()


def terminal_fallback():
    """Terminal mode fallback."""
    project_dir = get_project_dir()
    os.chdir(project_dir)
    
    if check_files():
        print("✓ Ready")
        webbrowser.open(APP_URL)
        python = find_python()
        subprocess.run([python, "app.py"])
    else:
        print("Setup required. Run in GUI mode.")


if __name__ == "__main__":
    try:
        gui_launch()
    except Exception as e:
        print(f"Error: {e}")
        terminal_fallback()