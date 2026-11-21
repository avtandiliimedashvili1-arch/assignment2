"""Terminal version of the Study & Task Planner Agent.  Run:  python main.py"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

if not os.getenv("GOOGLE_API_KEY"):
    sys.exit("GOOGLE_API_KEY is missing. Copy .env.example to .env and paste your key.")

from agent import chat_turn, create_agent, resume_turn  # noqa: E402

THREAD_ID = "cli-session"  # same id = the agent remembers the whole conversation


def show(kind, content, agent):
    """Print the reply. If approval is needed, ask the user until the agent finishes."""
    while kind == "approval":
        print(f"\n[APPROVAL NEEDED] {content['question']}")
        answer = input("Type yes or no: ").strip().lower()
        kind, content = resume_turn(agent, "yes" if answer in ("y", "yes") else "no", THREAD_ID)
    print(f"\nAgent: {content}\n")


def main():
    agent = create_agent()
    print("Study & Task Planner Agent. Type 'quit' to exit.\n")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        kind, content = chat_turn(agent, text, THREAD_ID)
        show(kind, content, agent)


if __name__ == "__main__":
    main()
