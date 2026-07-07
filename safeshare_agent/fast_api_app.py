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

import base64
import logging
import os
from typing import Any

import google.auth
import uvicorn
import vertexai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.auth.exceptions import DefaultCredentialsError
from google.genai import types
from pydantic import BaseModel

from safeshare_agent.agent import app as adk_app

# Load environment variables from .env file
load_dotenv()

# Telemetry: Set otel_to_cloud=False
os.environ["ADK_OTEL_TO_CLOUD"] = "0"
otel_to_cloud = False

# Set a fallback project ID if none is set to prevent crash on import
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    try:
        _, project_id = google.auth.default()
        if project_id:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        else:
            os.environ["GOOGLE_CLOUD_PROJECT"] = "dummy-project"
    except DefaultCredentialsError:
        os.environ["GOOGLE_CLOUD_PROJECT"] = "dummy-project"

if os.environ.get("GOOGLE_CLOUD_PROJECT") != "dummy-project":
    try:
        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    except Exception:
        pass

# Configure standard Python logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safeshare_agent_api")

app = FastAPI(title="SafeShare Agent Ambient Service")

# Initialize Session Service and Runner
session_service = InMemorySessionService()
runner = Runner(
    app=adk_app,
    session_service=session_service,
    auto_create_session=True,
)


class PubSubMessage(BaseModel):
    data: str | None = None
    messageId: str | None = None
    publishTime: str | None = None


class PubSubPayload(BaseModel):
    message: PubSubMessage
    subscription: str


class ConfirmRequest(BaseModel):
    session_id: str
    confirm: str


@app.post("/")
async def handle_pubsub_message(payload: PubSubPayload) -> Any:
    """Pub/Sub push endpoint that parses messages and triggers the workflow."""
    try:
        # 1. Normalize the subscription path to a short name
        subscription_path = payload.subscription
        short_name = (
            subscription_path.split("/")[-1] if subscription_path else "default-sub"
        )

        # 2. Decode the base64 data field
        if not payload.message.data:
            raise HTTPException(status_code=400, detail="No message data found.")

        decoded_bytes = base64.b64decode(payload.message.data)
        request_text = decoded_bytes.decode("utf-8").strip()
        logger.info(
            f"Received request from subscription '{short_name}': '{request_text}'"
        )

        # 3. Create or load session using the normalized subscription name
        session_id = f"session-{short_name}"

        # 4. Feed the payload into the runner
        new_message = types.Content(
            role="user", parts=[types.Part.from_text(text=request_text)]
        )

        events = list(
            runner.run(
                new_message=new_message,
                user_id=short_name,
                session_id=session_id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

        # Check if the execution paused on a RequestInput (Human-in-the-loop step)
        hitl_prompt = None
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if (
                        part.function_call
                        and part.function_call.name == "adk_request_input"
                    ):
                        hitl_prompt = (
                            part.function_call.args.get("message")
                            or "Awaiting confirmation"
                        )
                        break
                if hitl_prompt:
                    break

        if hitl_prompt:
            logger.info(
                f"Workflow paused for HIL confirmation. Session ID: {session_id}"
            )
            return {
                "status": "PAUSED",
                "session_id": session_id,
                "prompt": hitl_prompt.strip(),
            }

        # Otherwise, retrieve the final model response
        final_output = ""
        if events:
            for event in reversed(events):
                if event.content and event.content.parts:
                    final_output = "".join(
                        p.text for p in event.content.parts if p.text
                    )
                    if final_output:
                        break

        return {
            "status": "COMPLETED",
            "session_id": session_id,
            "output": final_output,
        }

    except Exception as e:
        logger.exception("Error processing Pub/Sub message")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/confirm")
async def confirm_split(req: ConfirmRequest) -> Any:
    """Submit human confirmation to resume a paused session."""
    try:
        session_id = req.session_id
        confirm_value = req.confirm.strip().lower()

        # Extract user_id from the session_id structure
        if session_id.startswith("session-"):
            user_id = session_id[len("session-") :]
        else:
            user_id = "default-user"

        logger.info(
            f"Resuming session '{session_id}' with confirmation: '{confirm_value}'"
        )

        # Resume the workflow by passing the function response in a Content payload
        resume_message = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="confirm",
                        id="confirm",
                        response={"confirm": confirm_value},
                    )
                )
            ],
        )

        events = list(
            runner.run(
                new_message=resume_message,
                user_id=user_id,
                session_id=session_id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

        # Retrieve the final commit or cancellation output
        final_output = ""
        if events:
            for event in reversed(events):
                if event.content and event.content.parts:
                    final_output = "".join(
                        p.text for p in event.content.parts if p.text
                    )
                    if final_output:
                        break

        return {
            "status": "COMPLETED",
            "session_id": session_id,
            "output": final_output,
        }

    except Exception as e:
        logger.exception("Error resuming session confirmation")
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run("safeshare_agent.fast_api_app:app", host="127.0.0.1", port=8080)
