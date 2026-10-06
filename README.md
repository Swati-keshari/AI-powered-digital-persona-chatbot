# Digital Persona: AI Chatbot with Tool Calling

A chatbot that answers questions about my career, skills and projects in my own voice, so recruiters and visitors can chat with my profile instead of reading it.

Built with **Python**, **Gradio** and an **LLM through Ollama** (any OpenAI-compatible endpoint works).

## How it works

1. `me/summary.txt` (and optionally a LinkedIn PDF export, `me/linkedin.pdf`) is loaded and placed in the **system prompt**.
2. The model answers as me and is told never to invent experience that isn't in the profile.
3. The model can call two **tools**, defined as JSON schemas:
   - `record_user_details`: when a visitor shares their email, it's recorded.
   - `record_unknown_question`: when the bot can't answer something, the question is logged so I can improve the profile.
4. A **tool-call loop** runs the requested tools, sends the results back to the model, and repeats until the model gives a final answer.
5. Each tool call sends a **Pushover** notification to my phone, if Pushover keys are set.

```
User ─► Gradio chat UI ─► LLM (system prompt + tools)
                              │
                 tool call? ──┤── yes ─► run tool ─► Pushover alert ─► result back to LLM
                              │
                              └── no ──► reply to user
```

## Tech stack

| Part | Technology |
|---|---|
| Language | Python |
| LLM | Qwen via Ollama, using the OpenAI-compatible SDK |
| UI | Gradio `ChatInterface` |
| PDF parsing | PyPDF |
| Notifications | Pushover API |

## Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env      # set the model endpoint and, optionally, Pushover keys
python app.py
```

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

## Project structure

```
app.py                  # the chatbot app (Gradio UI, tools, tool-call loop)
digital_persona.ipynb   # step-by-step notebook I used to build and test it
me/summary.txt          # profile the persona answers from
requirements.txt
.env.example
```

## Example questions

- "What projects have you worked on?"
- "What's your tech stack?"
- "Tell me about your internship."
- "Are you open to remote roles?"

## Next steps

- Deploy to Hugging Face Spaces
- Add RAG over a larger knowledge base about my work
- Add an evaluator model that checks answers before they're sent

## Credits

Built while learning agentic AI, based on the "Professionally You" lab from Ed Donner's Agentic AI course.

---

**Swati Keshari**: [GitHub](https://github.com/Swati-keshari) · [LinkedIn](https://www.linkedin.com/in/swati-keshari-65bb662bb/)
