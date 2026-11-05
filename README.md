# Payment Dispute Handler

Decides what to do with a customer's payment dispute, such as "I never received my order" or "I was charged twice".

It checks the transaction, shipping, refund and subscription records, applies the company rules, and picks one outcome: refund, deny, ask for evidence, or send to a human. Refunds over 500 USD and unclear cases always go to a human.

## Setup
1. Install Python 3.10 or newer.
2. In the project folder, run:

       pip install -e ".[dev]"

3. Get a free key from aistudio.google.com and set it (PowerShell):

       $env:GEMINI_API_KEY="your key"

## Usage
Create the test cases (240 made-up disputes):

    python -m dispute_agent.cli generate

Run one case and see the full result:

    python -m dispute_agent.cli demo CASE-0007 --model gemini:gemini-3.5-flash

Score it on a batch of cases (results go to the reports folder):

    python -m dispute_agent.cli eval --agents naive llm-v2 --limit 20 --model gemini:gemini-3.5-flash

Run the tests:

    pytest -q
