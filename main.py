"""
main.py
--------
Entry point for the ML Robot Training GUI application.
"""

import tkinter as tk
from gui.model_training_tab import ModelTrainingTab

def main():
    root = tk.Tk()
    root.title("ML Robot Training")
    root.geometry("800x600")
    
    # Create and pack the ModelTrainingTab frame
    training_tab = ModelTrainingTab(root)
    training_tab.pack(fill="both", expand=True)
    
    root.mainloop()

if __name__ == '__main__':
    main()
