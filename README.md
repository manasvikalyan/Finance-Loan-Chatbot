## Finance Loan Chatbot – How It Works


Structure of  main.py

- **Data loading**
  - On startup, the app reads `data.json` into memory as `loan_data`.
  - Each entry contains `customer_id`, `customer_name`, `total_due`, `due_date`, and `dpd`.

- **Tool functions (LangChain Tools)**
  - `get_customer_details(customer_id)`: returns the customer record from `loan_data` so Jiya can greet them by name.
  - `get_loan_details(customer_id, customer_name)`: returns the due amount and due date after the user confirms their identity.
  - `record_commitment(customer_id, commitment_date)`: records the promised payment date (mock implementation).

- **LLM and agent setup**
  - `init_chat_model("groq:llama-3.1-8b-instant", api_key=GROQ_API_KEY)` creates the LLM client.
  - A system prompt defines Jiya’s behavior (confirm identity → fetch loan details → ask for commitment).
  - `create_agent(llm, tools=..., system_prompt=...)` builds a LangGraph-based agent that can call the tools.

- **Session and memory handling**
  - The app keeps an in-memory `sessions` dictionary: `session_id -> list of messages`.
  - For each request, it loads the existing message history (if any), appends the new user message, runs the agent, and saves the updated history.

- **`POST /chat` endpoint behavior**
  - **Start a new call**: send `{ "customer_id": "...", "new_call": true }`.
    - The server creates a new `session_id`, adds an internal instruction, and the agent responds with Jiya’s first greeting.
  - **Continue a call**: send `{ "session_id": "...", "message": "user text" }`.
    - The server appends the message to that session’s history, runs the agent, and returns Jiya’s next reply.
  - The response always has:
    - `session_id`: to reuse on the next turn.
    - `reply`: Jiya’s latest message text.

- **Running the API**
  - Install dependencies: `pip install -r requirements.txt`
  - Start the server: `uvicorn main:app --reload`
  - Test in the browser at `http://127.0.0.1:8000/docs` or via curl:
    - New call:
      - `{"customer_id": "C456", "new_call": true}`
    - Next turn:
      - `{"session_id": "<from previous response>", "message": "Yes, speaking."}`


Steps to run the chat
1. 
{
  "customer_id": "C456",
  "new_call": true
}

2. 
{
  "session_id": "some-uuid",
  "message": "Yes, speaking."
}

3.
{
  "session_id": "some-uuid",
  "message": "I can pay tomorrow."
}

### Screenshots

![Screenshot 1](Screenshot%202025-11-24%20at%2010.43.29%E2%80%AFPM.png)

![Screenshot 2](Screenshot%202025-11-24%20at%2010.45.54%E2%80%AFPM.png)

![Screenshot 3](Screenshot%202025-11-24%20at%2010.48.19%E2%80%AFPM.png)

![Screenshot 4](Screenshot%202025-11-24%20at%2010.48.58%E2%80%AFPM.png)

![Screenshot 5](Screenshot%202025-11-24%20at%2010.49.10%E2%80%AFPM.png)

![Screenshot 6](Screenshot%202025-11-24%20at%2010.49.22%E2%80%AFPM.png)