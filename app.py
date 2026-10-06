"""Digital Persona: a chatbot that answers questions about Swati Keshari's career.

Profile data from me/ goes into the system prompt. The model can call two tools:
record_user_details (a visitor shares their email) and record_unknown_question
(a question the bot couldn't answer). Both send a Pushover notification if keys are set.
"""

import json
import os
from pathlib import Path

import gradio as gr
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(override=True)

ME_DIR = Path(__file__).parent / "me"
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
MAX_TOOL_ROUNDS = 5


def push(message):
    """Send a Pushover notification, or just print it if Pushover isn't configured."""
    print(f"Push: {message}", flush=True)
    user, token = os.getenv("PUSHOVER_USER"), os.getenv("PUSHOVER_TOKEN")
    if user and token:
        requests.post(PUSHOVER_URL, data={"user": user, "token": token, "message": message}, timeout=10)


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}


TOOL_FUNCTIONS = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The email address of this user"},
                    "name": {"type": "string", "description": "The user's name, if they provided it"},
                    "notes": {
                        "type": "string",
                        "description": "Any additional information about the conversation that's worth recording to give context",
                    },
                },
                "required": ["email"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question that couldn't be answered"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
]


class Me:
    def __init__(self):
        self.name = "Swati Keshari"
        self.model = os.getenv("MODEL", "qwen3.5:397b-cloud")
        self.client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        )
        self.summary = (ME_DIR / "summary.txt").read_text(encoding="utf-8")
        # Optional: a LinkedIn profile exported as PDF (LinkedIn > More > Save to PDF)
        self.linkedin = ""
        linkedin_pdf = ME_DIR / "linkedin.pdf"
        if linkedin_pdf.exists():
            for page in PdfReader(linkedin_pdf).pages:
                self.linkedin += page.extract_text() or ""

    def handle_tool_calls(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = TOOL_FUNCTIONS.get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        prompt = (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {self.name}'s background which you can use to answer questions. "
            "Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            "Never invent experience, skills or facts that are not in the information below. "
            "If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, "
            "even if it's about something trivial or unrelated to career. "
            "If the user is engaging in discussion, try to steer them towards getting in touch via email; "
            "ask for their email and record it using your record_user_details tool."
        )
        prompt += f"\n\n## Summary:\n{self.summary}\n\n"
        if self.linkedin:
            prompt += f"## LinkedIn Profile:\n{self.linkedin}\n\n"
        prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return prompt

    def chat(self, message, history):
        history = [{"role": h["role"], "content": h["content"]} for h in history]
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOLS)
            choice = response.choices[0]
            if choice.finish_reason != "tool_calls":
                return choice.message.content
            messages.append(choice.message)
            messages.extend(self.handle_tool_calls(choice.message.tool_calls))
        # Stop a model that keeps calling tools; answer once more without tools
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content


if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, title="Chat with Swati Keshari").launch()
