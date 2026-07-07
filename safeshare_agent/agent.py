# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import Agent as LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.models import Gemini
from google.adk.workflow import Edge, Workflow, node
from google.genai import types
from pydantic import BaseModel, Field

from safeshare_agent.config import MODEL_NAME
from safeshare_agent.database import (
    add_user_if_not_exists,
    get_known_users,
    init_db,
    insert_expense,
)

# Initialize and seed database on import
init_db()

# =====================================================================
# Pydantic Schemas for Structured I/O
# =====================================================================


class ExpenseSplit(BaseModel):
    user: str = Field(
        description="The user token (e.g., User_A, User_B) involved in the split."
    )
    amount: float = Field(
        description="The amount of money this user owes for the expense."
    )


class SplitResponse(BaseModel):
    description: str = Field(description="A brief description of the expense.")
    total_amount: float = Field(description="The total amount of the expense.")
    payer: str = Field(
        description="The user token who paid for the expense (e.g., User_A)."
    )
    splits: list[ExpenseSplit] = Field(
        description="The list of splits indicating how much each user owes."
    )


# =====================================================================
# Workflow Node Definitions
# =====================================================================


@node
def security_checkpoint(ctx: Context, node_input: types.Content) -> Event:
    """Node 1 (Security Masking & Prompt Injection Defense): Checkpoint for input processing."""
    # Extract text from input Content
    text = ""
    if isinstance(node_input, str):
        text = node_input
    elif hasattr(node_input, "parts") and node_input.parts:
        text = "".join(part.text for part in node_input.parts if part.text)

    # 1. Prompt Injection Defense
    injection_keywords = [
        "ignore rules",
        "ignore previous instructions",
        "auto-approve",
        "auto approve",
        "set balance",
        "drop table",
        "delete from",
        "insert into",
        "select *",
        "system instruction",
        "you are now",
        "new role",
        "prompt injection",
        "override",
    ]

    text_lower = text.lower()
    has_injection = any(keyword in text_lower for keyword in injection_keywords)

    if has_injection:
        return Event(
            output=text,
            actions=EventActions(
                state_delta={"security_violation": True, "violation_text": text},
                route="security_flagged",
            ),
        )

    # 2. Structural Data Masking (PII)
    known_users = get_known_users()
    current_user = "Tan"  # Default to Tan as current user

    # Initialize mappings
    token_mapping = {"User_A": current_user}
    reverse_mapping = {current_user.lower(): "User_A"}

    token_counter = ord("B")  # Start other tokens from User_B, User_C...

    # Map other known users in the database
    for user_name in known_users:
        if user_name.lower() == current_user.lower():
            continue
        # Use regex to find if name is mentioned in the text
        pattern = re.compile(rf"\b{re.escape(user_name)}\b", re.IGNORECASE)
        if pattern.search(text):
            token = f"User_{chr(token_counter)}"
            token_counter += 1
            token_mapping[token] = user_name
            reverse_mapping[user_name.lower()] = token

    # Perform replacements in text
    masked_text = text

    # Replace explicit names
    for name_lower, token in reverse_mapping.items():
        pattern = re.compile(rf"\b{re.escape(name_lower)}\b", re.IGNORECASE)
        masked_text = pattern.sub(token, masked_text)

    # Replace "I", "me", "my" references with the current user token (User_A)
    # only replacing complete words (case-insensitive)
    i_patterns = [
        (re.compile(r"\bI\b"), "User_A"),
        (re.compile(r"\bme\b", re.IGNORECASE), "User_A"),
        (re.compile(r"\bmy\b", re.IGNORECASE), "User_A"),
    ]
    for pattern, replacement in i_patterns:
        masked_text = pattern.sub(replacement, masked_text)

    # Return masked text and propagate mapping in the event state
    return Event(
        output=masked_text,
        actions=EventActions(
            state_delta={"token_mapping": token_mapping}, route="clean"
        ),
    )


# Node 2 (LLM Splitter): Passes sanitized text to model to extract splits
llm_splitter = LlmAgent(
    name="llm_splitter",
    model=Gemini(
        model=MODEL_NAME,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a precise expense splitter helper.\n"
        "You will receive a sanitized string where real names are masked as User_A, User_B, User_C, etc.\n"
        "Calculate the mathematical splits, identify the payer, and extract the expense description.\n"
        "You must follow the output schema exactly.\n"
        "Do NOT perform any external tool calls."
    ),
    output_schema=SplitResponse,
    output_key="split_details",
)


@node(rerun_on_resume=True)
async def ask_confirmation(
    ctx: Context, node_input: Any
) -> AsyncGenerator[Event | RequestInput, None]:
    """Node 3 (Human-in-the-loop): Pauses the workflow using RequestInput for confirmation."""
    # Check if a security violation was flagged
    if ctx.state.get("security_violation"):
        if not ctx.resume_inputs or "confirm" not in ctx.resume_inputs:
            violation_text = ctx.state.get("violation_text", "")
            message = (
                f"\n⚠️  [HIGH SEVERITY SECURITY ALERT] ⚠️\n"
                f"Adversarial prompt injection attempt detected!\n"
                f"Input text: '{violation_text}'\n"
                f"The request has been blocked and LLM processing bypassed.\n"
                f"Acknowledge and cancel the request? (y/n): "
            )
            yield RequestInput(interrupt_id="confirm", message=message)
            return

        # Once acknowledged/confirmed, route straight to canceled
        yield Event(
            output="Request aborted due to security violation.",
            actions=EventActions(route="canceled"),
        )
        return

    # Normal path
    # If the input was returned as a dictionary from LlmAgent, convert it to SplitResponse
    if isinstance(node_input, dict):
        split_data = SplitResponse(**node_input)
    else:
        split_data = node_input

    if not ctx.resume_inputs or "confirm" not in ctx.resume_inputs:
        mapping = ctx.state.get("token_mapping", {})

        def get_name(token: str) -> str:
            return mapping.get(token, token)

        splits_str = "\n".join(
            [
                f"  - {get_name(item.user)} owes ${item.amount:.2f}"
                for item in split_data.splits
            ]
        )

        message = (
            f"\n=== PROPOSED EXPENSE SPLIT ===\n"
            f"Description: {split_data.description}\n"
            f"Payer: {get_name(split_data.payer)}\n"
            f"Total: ${split_data.total_amount:.2f}\n"
            f"Splits:\n{splits_str}\n"
            f"Confirm and commit to database? (y/n): "
        )

        yield RequestInput(interrupt_id="confirm", message=message)
        return

    # User has responded, read from resume_inputs
    reply = ctx.resume_inputs["confirm"]
    if isinstance(reply, dict):
        reply_str = reply.get("confirm") or reply.get("response") or str(reply)
    else:
        reply_str = str(reply)
    reply_str = reply_str.strip().lower()
    if reply_str in ("y", "yes"):
        yield Event(output=split_data, actions=EventActions(route="confirmed"))
    else:
        yield Event(output="Canceled.", actions=EventActions(route="canceled"))


@node
def db_commit(ctx: Context, node_input: SplitResponse) -> Event:
    """Node 4 (DB Commit): Unmasks tokens, maps to User IDs, and inserts into 3NF SQLite."""
    mapping = ctx.state.get("token_mapping", {})

    # Resolve payer ID (create if not exists)
    payer_name = mapping.get(node_input.payer, node_input.payer)
    payer_id = add_user_if_not_exists(payer_name)

    # Resolve split user IDs
    resolved_splits = []
    for split in node_input.splits:
        user_name = mapping.get(split.user, split.user)
        user_id = add_user_if_not_exists(user_name)
        resolved_splits.append((user_id, split.amount))

    # Insert expense and splits transactionally
    expense_id = insert_expense(
        description=node_input.description,
        total_amount=node_input.total_amount,
        payer_id=payer_id,
        splits=resolved_splits,
    )

    msg = (
        f"Successfully committed expense ID {expense_id} ('{node_input.description}') "
        f"of ${node_input.total_amount:.2f} paid by {payer_name} to the SQLite database."
    )

    # Emit both output (internal) and content (renders in playground/CLI)
    return Event(
        output=msg,
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
    )


@node
def cancel_split(node_input: str) -> Event:
    """Node 5 (Cancel Handler): Responds to canceled split request."""
    msg = f"Expense split canceled: {node_input}"
    return Event(
        output=msg,
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
    )


# =====================================================================
# ADK 2.0 Graph Workflow Definition
# =====================================================================

root_agent = Workflow(
    name="safeshare_workflow",
    edges=[
        ("START", security_checkpoint),
        Edge(from_node=security_checkpoint, to_node=llm_splitter, route="clean"),
        (llm_splitter, ask_confirmation),
        Edge(
            from_node=security_checkpoint,
            to_node=ask_confirmation,
            route="security_flagged",
        ),
        Edge(from_node=ask_confirmation, to_node=db_commit, route="confirmed"),
        Edge(from_node=ask_confirmation, to_node=cancel_split, route="canceled"),
    ],
    description="Privacy-preserving expense splitter using PII masking and human confirmation.",
)

# App configuration with resumability enabled (required for HITL)
app = App(
    root_agent=root_agent,
    name="safeshare_agent",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
