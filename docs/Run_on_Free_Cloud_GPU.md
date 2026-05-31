# How to Run on Free Cloud GPU

Since you do not have a local GPU, it is highly recommended to use **Google Colab** or **Kaggle Kernels**. Both provide free access to GPUs (like NVIDIA T4).

## Option 1: Google Colab (Recommended)

1. **Open Google Colab**: Go to [https://colab.research.google.com/](https://colab.research.google.com/).
2. **Create a New Notebook**: Click on "New Notebook".
3. **Enable GPU**:
   - Go to the menu: `Runtime` > `Change runtime type`.
   - Under "Hardware accelerator", select **T4 GPU**.
   - Click "Save".
4. **Upload Files**:
   - Click on the folder icon on the left sidebar ("Files").
   - Click the "Upload" button (file with an arrow).
   - Upload the `solution.py` file.
   - *Note:* You do **NOT** need to manually download/upload the English-Urdu dataset. The script will try to download it automatically.
5. **Run the Code**:
   - In the first cell of the notebook, run the following commands:
     ```python
     !pip install kagglehub openpyxl nltk
     !python solution.py
     ```
   - Alternatively, you can open `solution.py` in a text editor (Notepad), copy the entire code, and paste it into a code cell in Colab, then press Shift+Enter to run it.

## Option 2: Using Kaggle

1. Go to [https://www.kaggle.com/code](https://www.kaggle.com/code).
2. Click "New Notebook".
3. In the right sidebar, under "Session Options", set "Accelerator" to **GPU T4 x2**.
4. To add data (for Q1), click "Add Input" and search for "English Urdu" or upload your `english_urdu.csv`.
5. Copy-paste the code from `solution.py` into a cell and run it.

## Important Notes

- **Dataset for Q1**: The code expects a file named `english_urdu.csv` in the same directory. If you are using the Kaggle dataset, you might need to adjust the path in the code (e.g., `/kaggle/input/...`) or rename the file after uploading.
- **Dependencies**: The script intentionally uses standard libraries. If you encounter `ModuleNotFoundError` for `nltk`, run `!pip install nltk` in a cell before running the script.
- **Output**: The script will print loss values and save visualization images (`q2_results.png`, `q3_reconstruction.png`, etc.) in the current directory. You can view these in the Files sidebar.
