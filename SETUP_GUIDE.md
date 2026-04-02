# AskLex Legal Chatbot – Setup & Run Guide

This guide explains how to install everything and run your **text + voice** legal chatbot.

---

## 1. Project Files

Your project folder should contain:

- `main.py`  → the only Python file you run (chatbot + voice)
- `requirements.txt` → list of Python dependencies
- `.env` → stores your Cerebras API key
- `index.html` (optional) → web UI (not required to run the chatbot)

---

## 2. Create / Check `.env`

1. Open the file named `.env` in your project.
2. Put your Cerebras API key like this:

```text
CEREBRAS_API_KEY=sk-your_real_key_here
```

3. Save the file.

> Without this key, the chatbot will still run but use a simple fallback answer instead of Cerebras.

---

## 3. Install Dependencies

You only need to do this once per machine / virtual environment.

### 3.1. Open a terminal in your project folder

In VS Code:
- Open the folder with `main.py`.
- Open **Terminal → New Terminal**.

OR in normal terminal:

```bash
cd /path/to/your/project
```

### 3.2. (Optional but recommended) Create a virtual environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3.3. Install all Python packages

```bash
pip install -r requirements.txt
```

If you see errors about `pyaudio` on Windows, you can either:
- Use a prebuilt wheel for PyAudio, OR
- Skip microphone input (text + voice output will still work).

---

## 4. Run the Chatbot in VS Code / Local Terminal

1. Make sure your terminal is inside the project folder and (optionally) the virtual environment is activated.
2. Run:

```bash
python main.py
```

3. You will see something like:

```text
======================================================================
  AskLex: Bangladesh Legal AI Chatbot
======================================================================

Environment:
  GPU: True/False
  Voice Input: True/False
  Voice Output: True/False
  Colab: False

[INFO] Loading dataset...
[INFO] Building indexes...

======================================================================
Commands:
  Ask: Type your legal question
  'record': Use microphone (local only)
  'profile': Set user profile (religion, marital status, etc.)
  'history': Show conversation
  'exit': Quit
======================================================================

You:
```

---

## 5. Using the Chatbot – Text & Voice

### 5.1. Ask a text question

At the `You:` prompt, type a question, for example:

```text
You: What are my rights in a dowry case?
```

The bot will:
- Retrieve legal sections from the UKIL dataset.
- Call Cerebras (if the key is set).
- Print:

```text
Assistant: [detailed legal answer...]
```

If voice output is available, it will also say the answer aloud.

### 5.2. Use microphone input (voice → text)

1. At the `You:` prompt, type:

```text
You: record
```

2. Speak your question when instructed (for ~5–10 seconds).
3. The program will show something like:

```text
Recording...
Recognized: What are my rights in a dowry case?
Processing...
Assistant: [answer]
```

If speech recognition fails, it will tell you to type instead.

### 5.3. Set user profile (optional)

Profile helps tailor answers (e.g., religion, marital status).

1. At the prompt, type:

```text
You: profile
```

2. Then enter key–value pairs, for example:

```text
religion: Muslim, marital_status: married
```

3. The chatbot will confirm:

```text
✓ Profile updated
```

### 5.4. View conversation history

```text
You: history
```

The program prints your recent questions and answers.

### 5.5. Exit the program

```text
You: exit
```

OR press `Ctrl + C` once in the terminal.

---

## 6. Run in Google Colab (Text Only)

Voice input/output usually does **not** work in Colab. Use text chat only.

1. Upload these files to Colab:
   - `main.py`
   - `requirements.txt` (optional)

2. Install required packages in a notebook cell:

```python
!pip install pandas numpy networkx nltk torch sentence-transformers rank-bm25
!pip install openai python-dotenv
import nltk
nltk.download("punkt")
```

3. Set your API key in a cell:

```python
import os
os.environ["CEREBRAS_API_KEY"] = "sk-your_real_key_here"
```

4. Run the chatbot script (CLI inside Colab):

```python
!python main.py
```

5. Use text commands exactly like on your local machine:

- Type questions when you see `You:`
- Use `profile`, `history`, `exit`
- Do **not** use `record` in Colab.

---

## 7. Quick Troubleshooting

- **`CEREBRAS_API_KEY not set` warning:**
  - Check `.env` or environment variable.
  - Make sure there is no extra space around the key.

- **Import errors (e.g., sentence_transformers, rank_bm25):**
  - Re-run: `pip install -r requirements.txt`
  - Make sure you’re in the correct virtual environment.

- **PyAudio install fails (Windows):**
  - You can still use:
    - Text questions
    - Printed answers
    - Voice output (if `pyttsx3` installs correctly)
  - Just avoid the `record` command.

---

You now have a single, clean setup:
- Run `python main.py` to start.
- Use text and (when available) voice in the same program.
