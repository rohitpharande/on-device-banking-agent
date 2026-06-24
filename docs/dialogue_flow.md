# Dialogue Flow & State Management

The core design principle of the **On-Device Banking Agent** is to separate **language understanding** (handled by the fine-tuned Small Language Model) from **dialogue state management** (handled deterministically by Python code). This design completely eliminates hallucination risk, ensures precise API calls, and supports natural conversational fallback.

---

## 🧭 Architecture Flow

```
   User Voice / Text Utterance
                │
                ▼
   ┌───────────────────────────┐
   │    Speech Recognition     │ (Faster Whisper)
   └────────────┬──────────────┘
                │
                ▼ Transcript
   ┌───────────────────────────┐
   │    Intent & Slot Model    │ (Fine-tuned Qwen3-0.6B)
   └────────────┬──────────────┘
                │
                ▼ Structured JSON (Intent + Slots)
   ┌───────────────────────────┐
   │    Dialogue Orchestrator  │ (Deterministic State Machine)
   └────────────┬──────────────┘
                │
         ┌──────┴──────────────┐
         │ All Slots Filled?   │
         └──┬───────────────┬──┘
            │ Yes           │ No
            ▼               ▼
   ┌─────────────────┐ ┌───────────────────────────┐
   │ Execute Backend │ │ Elicit Next Missing Slot  │
   │ & Respond       │ │ (e.g. "which account?")   │
   └─────────────────┘ └───────────────────────────┘
```

---

## 🧠 Structured Output Format

The fine-tuned Qwen3 model is trained to output a single JSON block representing the user's intent and any extracted slots (arguments).

#### Sample Input Transcript:
> *"I need to transfer four hundred dollars from savings to my checking account"*

#### Model JSON Output:
```json
{
  "name": "transfer_money",
  "arguments": {
    "amount": 400.0,
    "from_account": "savings",
    "to_account": "checking"
  }
}
```

---

## 🔄 State Machine & Slot Elicitation

The dialogue is managed by the `TextOrchestrator` class in `src/orchestrator_qwen.py`. 

### 1. Intent Locking
When a user initiates an action requiring slots (e.g., `transfer_money` or `pay_bill`), the orchestrator locks this as the `active_intent`. 

### 2. Conversational Intent Filtering
While a transaction is active, the orchestrator filters out generic conversational intents (e.g., `greeting` or `thank_you`). This prevents a casual greeting like *"oh thanks"* from clearing the session context while we are trying to collect an account type or amount.

### 3. Slot Merging
As the user provides more information across turns, the orchestrator merges new slots into `accumulated_slots`:
- **Turn 1**: *"Move $200"* → `{"amount": 200}`
- **Turn 2**: *"from checking"* → `{"amount": 200, "from_account": "checking"}`
- **Turn 3**: *"to savings"* → `{"amount": 200, "from_account": "checking", "to_account": "savings"}`

### 4. Slot Elicitation
If any required slots are missing, the orchestrator maps the missing arguments to human-readable prompts from `INDIVIDUAL_SLOT_PROMPTS` in `src/config.py` and returns a question to the user:
- Missing `to_account` → *"Could you provide which account to transfer to?"*

---

## 🛠️ Supported Intents & Slot Configurations

Here is how each supported intent maps to required slot parameters inside `src/config.py`:

| Intent Name | Required Arguments | Simulated Action / Template |
|---|---|---|
| `check_balance` | `account_type` | Returns a randomized float balance |
| `get_statement` | `account_type`, `period` | Simulates sending an email statement |
| `transfer_money` | `amount`, `from_account`, `to_account` | Simulates a local funds transfer |
| `pay_bill` | `payee`, `amount` | Simulates a bill payment |
| `cancel_card` | `card_type`, `card_last_four` | Simulates card deactivation |
| `replace_card` | `card_type`, `card_last_four` | Simulates mail order replacement |
| `activate_card` | `card_last_four` | Simulates card activation |
| `reset_pin` | `card_type`, `card_last_four` | Simulates mailing a new PIN code |
| `report_fraud` | `card_type`, `card_last_four`, `transaction_amount` | Flags card for fraud review |
| `speak_to_human` | *(none)* | Immediately routes to human queue |
| `greeting` | *(none)* | Returns generic hello response |
| `goodbye` | *(none)* | Ends the session |
| `thank_you` | *(none)* | Returns acknowledgment |
| `intent_unclear` | *(none)* | Re-lists assistant capabilities |
