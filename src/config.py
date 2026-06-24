"""Centralized configuration for the BankCo Voice Assistant.

Contains all tool definitions, slot-filling rules, response templates,
and the system prompt used by the fine-tuned Qwen3-0.6B model.
"""

# ---------------------------------------------------------------------------
# System prompt — must match the prompt used during fine-tuning
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a low-latency banking voice assistant. Analyze the "
        "conversation history and output ONLY a valid JSON object matching "
        "the required tool call structure."
    ),
}

# ---------------------------------------------------------------------------
# Tool definitions (14 banking functions)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_balance",
            "description": "Check the balance of a bank account",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings", "credit"],
                        "description": "Type of account to check balance for",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_statement",
            "description": "Request an account statement to be sent to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings", "credit"],
                        "description": "Type of account to get statement for",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["last_month", "last_3_months", "last_year"],
                        "description": "Time period for the statement",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_money",
            "description": "Transfer money between the user's own bank accounts",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Amount to transfer in dollars",
                    },
                    "from_account": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Account to transfer money from",
                    },
                    "to_account": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Account to transfer money to",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_card",
            "description": "Cancel and deactivate a bank card",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_type": {
                        "type": "string",
                        "enum": ["credit", "debit"],
                        "description": "Type of card to cancel",
                    },
                    "card_last_four": {
                        "type": "string",
                        "description": "Last 4 digits of the card number",
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["lost", "stolen", "damaged", "other"],
                        "description": "Reason for cancelling the card",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_card",
            "description": "Request a replacement card to be sent to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_type": {
                        "type": "string",
                        "enum": ["credit", "debit"],
                        "description": "Type of card to replace",
                    },
                    "card_last_four": {
                        "type": "string",
                        "description": "Last 4 digits of the card number",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_card",
            "description": "Activate a new card that was received in the mail",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_last_four": {
                        "type": "string",
                        "description": "Last 4 digits of the card number to activate",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_fraud",
            "description": "Report a fraudulent or unauthorized transaction on a card",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_type": {
                        "type": "string",
                        "enum": ["credit", "debit"],
                        "description": "Type of card with fraudulent activity",
                    },
                    "card_last_four": {
                        "type": "string",
                        "description": "Last 4 digits of the card number",
                    },
                    "transaction_amount": {
                        "type": "number",
                        "description": "Amount of the fraudulent transaction in dollars",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_pin",
            "description": "Reset the PIN for a bank card",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_type": {
                        "type": "string",
                        "enum": ["credit", "debit"],
                        "description": "Type of card to reset PIN for",
                    },
                    "card_last_four": {
                        "type": "string",
                        "description": "Last 4 digits of the card number",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pay_bill",
            "description": "Pay a bill to a company or service provider",
            "parameters": {
                "type": "object",
                "properties": {
                    "payee": {
                        "type": "string",
                        "description": "Name of the company or person to pay",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount to pay in dollars",
                    },
                    "from_account": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Account to pay from",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak_to_human",
            "description": "Connect the user to a human customer service agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "enum": ["general", "fraud", "loans", "technical"],
                        "description": "Department to connect to",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "intent_unclear",
            "description": "Use when the user's intent cannot be determined from their message",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "greeting",
            "description": "User is greeting or starting the conversation",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "goodbye",
            "description": "User is ending the conversation",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thank_you",
            "description": "User is expressing gratitude or thanks",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Required arguments per function (used by the slot-filling engine)
# ---------------------------------------------------------------------------
FUNCTION_REQUIRED_ARGS: dict[str, list[str]] = {
    "cancel_card": ["card_type", "card_last_four"],
    "replace_card": ["card_type", "card_last_four"],
    "activate_card": ["card_last_four"],
    "reset_pin": ["card_type", "card_last_four"],
    "transfer_money": ["amount", "from_account", "to_account"],
    "check_balance": ["account_type"],
    "pay_bill": ["payee", "amount"],
    "get_statement": ["account_type", "period"],
    "report_fraud": ["card_type", "card_last_four", "transaction_amount"],
    "speak_to_human": [],
    "greeting": [],
    "goodbye": [],
    "thank_you": [],
    "intent_unclear": [],
}

# ---------------------------------------------------------------------------
# Human-readable prompts for each missing slot (used during elicitation)
# ---------------------------------------------------------------------------
INDIVIDUAL_SLOT_PROMPTS: dict[str, dict[str, str]] = {
    "cancel_card": {
        "card_type": "credit or debit",
        "card_last_four": "the last 4 digits",
        "reason": "the reason for cancellation",
    },
    "replace_card": {
        "card_type": "credit or debit",
        "card_last_four": "the last 4 digits",
    },
    "activate_card": {
        "card_last_four": "the last 4 digits of the card",
    },
    "reset_pin": {
        "card_type": "credit or debit",
        "card_last_four": "the last 4 digits",
    },
    "transfer_money": {
        "amount": "the amount",
        "from_account": "which account to transfer from",
        "to_account": "which account to transfer to",
    },
    "check_balance": {
        "account_type": "the account type (checking, savings, or credit)",
    },
    "pay_bill": {
        "payee": "who to pay",
        "amount": "the amount",
    },
    "get_statement": {
        "account_type": "the account type (checking, savings, or credit)",
        "period": "the period for which to get the statement",
    },
    "report_fraud": {
        "card_type": "credit or debit",
        "card_last_four": "the last 4 digits",
        "transaction_amount": "the amount of the suspicious transaction",
    },
}

# ---------------------------------------------------------------------------
# Response templates — rendered after successful tool execution
# ---------------------------------------------------------------------------
SUCCESS_TEMPLATES: dict[str, str] = {
    "cancel_card": "Done. Your {card_type} card ending in {card_last_four} has been cancelled.",
    "replace_card": "I have placed a replacement card request for you. A new {card_type} card will arrive in 5-7 business days.",
    "activate_card": "Your card ending in {card_last_four} is now active.",
    "reset_pin": "Your PIN has been reset. You'll receive a new PIN by mail in 1-2 days.",
    "transfer_money": "Ok.I am initiating your transaction request. Please wait. Transaction is in progress. Transferred ${amount:.2f} from {from_account} to {to_account}.",
    "check_balance": "Your {account_type} balance is ${balance:.2f}.",
    "pay_bill": "Please wait. Processing your payment request. Paid ${amount:.2f} to {payee}.",
    "get_statement": "I'm sending your {account_type} statement to your registered email.",
    "report_fraud": "I've flagged your {card_type} card for review. Our fraud team will contact you within 24 hours.",
    "speak_to_human": "Connecting you to an agent now. Please hold.",
    "greeting": "Hello! Welcome to DemoBank. How can I help you today?",
    "goodbye": "Goodbye! Thanks for calling DemoBank.",
    "thank_you": "You're welcome! Is there anything else I can help with?",
}
