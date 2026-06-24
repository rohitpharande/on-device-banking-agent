# Training Data Format

Each line in a training JSONL file is a JSON object with a single `messages` key:

```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "tool_calls": [{"id": "call_X", "type": "function", "function": {"name": "greeting", "arguments": {}}}], "content": ""},
    {"role": "user", "content": "What is my balance?"},
    {"role": "assistant", "tool_calls": [{"id": "call_Y", "type": "function", "function": {"name": "check_balance", "arguments": {"account_type": "checking"}}}], "content": ""}
  ]
}
```

## Fields

- **messages**: Array of conversation turns
  - **user turns**: `role: "user"`, `content: "<speech transcript>"`
  - **assistant turns**: `role: "assistant"`, `tool_calls` array with a single function call, `content: ""`

## Available Functions

Refer to `src/config.py` for the full list of 14 banking functions including `check_balance`, `transfer_money`, `cancel_card`, `replace_card`, `report_fraud`, `pay_bill`, `speak_to_human`, and conversational intents (`greeting`, `goodbye`, `thank_you`, `intent_unclear`).

## Files

- `sample_train_data.jsonl` — 5 example rows to understand the format
- `train_data_original.jsonl` — Full training set (not tracked in git, see notebooks to generate)
- `train_data_rand.jsonl` — Randomized version with anonymized numbers

## Data Preparation
To generate the randomized training data from the base set refer to appendix in ./training/finetune_qwen3.ipynb
