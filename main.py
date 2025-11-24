"""
Minimal debt collections chatbot API using FastAPI + LangChain.

Single POST /chat endpoint that:
- Starts an outbound call for a given customer_id
- Maintains chat history in memory via session_id
- Uses LangChain tools + agent to drive the conversation
"""

import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# -----------------------------
# Data loading
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.json"

with DATA_PATH.open() as f:
    loan_data: Dict[str, dict] = json.load(f)


# -----------------------------
# Tools
# -----------------------------

@tool
def get_customer_details(customer_id: str) -> dict:
    """
    Fetch details for a customer ID from loan_data.
    Used to greet the customer by name:
    "Hello, this is Jiya calling from ABC Finance. Am I speaking with {customer_name}?"
    """
    return loan_data.get(customer_id, {"error": "Customer ID not found"})


@tool
def get_loan_details(customer_id: str, customer_name: str) -> dict:
    """
    After user confirms identity, fetch loan details and total due.
    The agent should say:
    "Thank you for confirming, {customer_name}. Your loan amount of rupees {total_due}
    is pending from {due_date}. When can you make the payment?"
    """
    record = loan_data.get(customer_id)
    if not record or record.get("customer_name") != customer_name:
        return {"error": "No due amount found."}
    return {
        "customer_name": record["customer_name"],
        "total_due": record["total_due"],
        "due_date": record["due_date"],
        "dpd": record.get("dpd", 0),
    }


@tool
def record_commitment(customer_id: str, commitment_date: str) -> str:
    """
    Record the user's payment commitment date.
    The agent should then thank the customer and repeat the commitment date.
    """
    # In a real system this would write to a DB or CRM.
    return f"Commitment for {customer_id} noted."


# -----------------------------
# LLM + Agent
# -----------------------------

# Load environment variables from .env if present
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set. "
        "Create a .env file with GROQ_API_KEY=your_key or export it in your shell."
    )

llm = init_chat_model("groq:llama-3.1-8b-instant", api_key=GROQ_API_KEY)

prompt = """
You are Jiya from ABC Finance. You MUST follow these conversation rules:

1. Step 1: Start by confirming you are speaking with the correct customer.
   Use get_customer_details with the provided customer_id to look up their name,
   then ask: "Hello, this is Jiya calling from ABC Finance. Am I speaking with {customer_name}?"

2. Step 2: Wait for the user to clearly confirm (e.g. "yes", "yeah", "speaking") before using get_loan_details.

3. Step 3: After confirmation, fetch loan details using get_loan_details.
   Then say politely: "Thank you for confirming, {customer_name}. Your loan amount of rupees {total_due}
   is pending from {due_date}. When can you make the payment?"

4. Step 4: When the user provides a payment date, call ONLY the record_commitment tool in that turn.
   After the tool result is returned, in the next message confirm the commitment and thank the customer.

5. Chat must always be polite, compliant, and concise.  If the user asks for any other information, provide the information that is available.

IMPORTANT: When calling any tool, your assistant response for that turn must contain ONLY the tool call.
"""

tools = [get_customer_details, get_loan_details, record_commitment]
agent = create_agent(llm, tools=tools, system_prompt=prompt)


# -----------------------------
# In-memory session store
# -----------------------------

SessionHistory = List[BaseMessage]
sessions: Dict[str, SessionHistory] = {}


def run_agent_with_history(history: SessionHistory) -> SessionHistory:
    """Helper to run agent with given history and return updated messages."""
    state = agent.invoke({"messages": history})
    return state["messages"]


def extract_last_ai_reply(history: SessionHistory) -> str:
    """Return the content of the last AI message in history."""
    for msg in reversed(history):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return "I'm sorry, I couldn't generate a response."


# -----------------------------
# FastAPI models & app
# -----------------------------


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    message: Optional[str] = None
    new_call: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str


app = FastAPI(title="Finance Loan Chatbot API")


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Single chat endpoint.

    - Start a new call: POST with new_call=true and customer_id set.
    - Continue a call: POST with existing session_id and message.
    """
    # Determine session
    session_id = payload.session_id or str(uuid.uuid4())
    history: SessionHistory = sessions.get(session_id, [])

    # Starting a new call
    if payload.new_call or not history:
        if not payload.customer_id:
            raise HTTPException(status_code=400, detail="customer_id is required to start a new call.")

        history = []
        # Internal instruction to agent (not shown to customer directly)
        init_text = (
            f"Start an outbound collection call for customer id {payload.customer_id}. "
            "First, look up their details using tools if needed, then say: "
            "'Hello, this is Jiya calling from ABC Finance. Am I speaking with <customer_name>?'"
        )
        history.append(HumanMessage(content=init_text))

    # Continuing an existing conversation
    elif payload.message:
        history.append(HumanMessage(content=payload.message))

    else:
        raise HTTPException(status_code=400, detail="Either new_call must be true or a message must be provided.")

    # Run agent and update session history
    history = run_agent_with_history(history)
    sessions[session_id] = history

    reply = extract_last_ai_reply(history)
    return ChatResponse(session_id=session_id, reply=reply)


# Optional: simple root check
@app.get("/")
def root() -> dict:
    return {"status": "ok", "message": "Finance Loan Chatbot API is running"}
