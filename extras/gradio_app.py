"""Minimal Gradio interface for the Amadeus assistant."""
from __future__ import annotations

import gradio as gr

from amadeus.config import Config
from amadeus.persona import build_system_prompt
from amadeus.providers.llm.base import ChatMessage
from amadeus.providers.llm.openai_provider import OpenAIProvider


CONFIG = Config.load()
LLM = OpenAIProvider(CONFIG.llm_model, api_key=CONFIG.openai_api_key)
SYSTEM_PROMPT = build_system_prompt(CONFIG.persona)


def chat(history: list[tuple[str, str]], message: str):
    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    for user, assistant in history:
        messages.append(ChatMessage(role="user", content=user))
        messages.append(ChatMessage(role="assistant", content=assistant))
    messages.append(ChatMessage(role="user", content=message))
    response = LLM.generate(messages)
    history.append((message, response))
    return history, ""


with gr.Blocks() as demo:
    gr.Markdown("# Amadeus — Kurisu Assistant")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Message", placeholder="Ask Kurisu something")
    clear = gr.Button("Clear")

    msg.submit(chat, [chatbot, msg], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
