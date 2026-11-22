# Study & Task Planner Agent

An AI agent that helps a student manage a to-do list, check the weather and look up topics on Wikipedia. It uses **LangGraph** with **Google Gemini**, remembers the conversation, and **asks for human approval before any destructive action** (deleting a task or clearing the whole list).

## Features
- **Tools (7):** `add_task`, `list_tasks`, `complete_task`, `delete_task`*, `clear_all_tasks`*, `get_weather`, `search_wikipedia` (*needs approval)
- **External APIs (no key needed):** Open-Meteo (weather), Wikipedia
- **Memory:** `MemorySaver` + `thread_id` keeps the whole conversation
- **StateGraph:** `assistant` -> `safe_tools` / `approval` -> `assistant`
- **Approval flow:** the graph pauses with `interrupt()`; the human answers yes or no

## Setup
```bash
git clone <your-repo-link>
cd <repo-folder>
pip install -r requirements.txt
```
Get a free Gemini API key at https://aistudio.google.com ("Get API key"), then copy `.env.example` to `.env` and fill it in:
```
GOOGLE_API_KEY=""
```
Never upload your real key to GitHub (`.env` is in `.gitignore`).

## How to run
| Version | Command |
|---|---|
| Terminal chat | `python main.py` |
| Streamlit chat UI | `streamlit run app.py` |
| Notebook | open `Practical_Assistant_Agent.ipynb` in Google Colab (add `GOOGLE_API_KEY` in Colab Secrets) |

## Example
```
You: add a task: finish the marketing report
You: delete task 1
[APPROVAL NEEDED] Delete task #1 ('finish the marketing report')?
Type yes or no: no
Agent: Okay, nothing was changed.
```

## Files
- `agent.py` - tools, state, graph (shared by everything)
- `main.py` - terminal version
- `app.py` - Streamlit version
- `Practical_Assistant_Agent.ipynb` - notebook with tests, approve/deny demos and eval table

## Notes
- Tasks are stored in memory, so they reset when the program restarts.
- If the model name is rejected, set `GEMINI_MODEL` in `.env` to a current model name from Google AI Studio.
