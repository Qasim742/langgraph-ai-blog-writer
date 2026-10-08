# Blog Writer Agent

A LangGraph-based blog-writing agent that uses Groq to plan and write a blog post, can use Tavily for web research, and can use Gemini to generate images.

## 1. Requirements

- Python 3.10 or newer
- A Groq API key (required)
- A Tavily API key (optional; enables web research)
- A Google Gemini API key (optional; enables image generation)

## 2. Download or clone the project

Place `bwa-research.py`, `requirements.txt`, and this `README.md` in the same project folder.

Open PowerShell in that folder.

## 3. Create and activate a virtual environment

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this command in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Configure API keys

Create a file named `.env` in the same folder as `bwa-research.py` and add your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

- `GROQ_API_KEY` is required for the agent to run.
- `TAVILY_API_KEY` is optional. Without it, the research function returns no search results.
- `GOOGLE_API_KEY` is optional until the agent attempts to generate images. Image generation can incur separate API usage or charges depending on your Google account and API plan.

Get keys from:

- Groq: https://console.groq.com/keys
- Tavily: https://app.tavily.com/home
- Google AI Studio: https://aistudio.google.com/apikey

Do not share your `.env` file or commit it to GitHub.

## 6. Run the agent

With the virtual environment activated and `.env` configured, run:

```powershell
python bwa-research.py
```

The script currently compiles the LangGraph application at the end of the file. If your local file only defines `app` and does not invoke it, add or retain your own input/invocation code to run a blog-writing task.

## 7. Troubleshooting

### `GROQ_API_KEY not found`

Make sure `.env` is in the same folder where you run the script and that the key is written as `GROQ_API_KEY=...` without quotes or spaces around `=`.

### Groq `429` / token-per-minute limit

Wait for the retry window to pass. If your code has retry handling enabled, it may retry automatically. Parallel workers can still exceed the account's TPM limit; lower the number of parallel requests or reduce prompt/output token usage if the issue continues.

### Research returns no results

Check that `TAVILY_API_KEY` is set correctly. The code currently catches research exceptions and returns an empty result list, so search errors may not stop the whole workflow.

### Gemini image generation fails

Check `GOOGLE_API_KEY`, confirm the selected image model is available to your API project, and check its current quotas and pricing. The Gemini app/student subscription does not necessarily cover API usage.

## 8. Keep API keys private

Add the following to `.gitignore` if it is not already present:

```gitignore
.env
venv/
__pycache__/
images/
```
