from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import json
from openai import AsyncOpenAI
from openai.types.beta.assistant_stream_event import ThreadMessageDelta, ThreadMessageCreated, ThreadRunCompleted
import os
from dotenv import load_dotenv
import asyncio
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get the assistant IDs from environment variables
ASSISTANT_ID_JOHNS_CREEK = os.getenv("ASSISTANT_ID_JOHNS_CREEK")
ASSISTANT_ID_ATLANTA = os.getenv("ASSISTANT_ID_ATLANTA")

# Semaphore to limit concurrent API calls
API_SEMAPHORE = asyncio.Semaphore(1000)  # Adjust this number based on your API rate limits

async def get_assistant_id(assistant_type: str):
    if assistant_type == "johns_creek":
        return ASSISTANT_ID_JOHNS_CREEK
    elif assistant_type == "atlanta":
        return ASSISTANT_ID_ATLANTA
    else:
        raise ValueError("Invalid assistant type")

class UserMessage(BaseModel):
    message: str

class ThreadMessage(BaseModel):
    message: str
    thread_id: str


@app.post("/chat")
async def chat_with_assistant(user_message: UserMessage, x_assistant_type: Optional[str] = Header(None)):
    return StreamingResponse(process_chat(user_message.message, assistant_type=x_assistant_type), media_type="text/event-stream")

@app.post("/chat_existing_thread")
async def chat_with_existing_thread(thread_message: ThreadMessage, x_assistant_type: Optional[str] = Header(None)):
    return StreamingResponse(process_chat(thread_message.message, thread_message.thread_id, assistant_type=x_assistant_type), media_type="text/event-stream")

async def process_chat(message: str, thread_id: str = None, assistant_type: str = None):
    try:
        if not message.strip():
            logger.warning("Received empty message, skipping processing.")
            yield "data: {}\n\n"
            return

        if thread_id is None:
            # Create a new thread
            async with API_SEMAPHORE:
                thread = await client.beta.threads.create()
                thread_id = thread.id
        
        logger.debug(f"Thread ID: {thread_id}")
        
        # Add the user's message to the thread
        async with API_SEMAPHORE:
            await client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message
            )

        logger.debug(f"Assistant type: {assistant_type}")
        # Get the appropriate assistant ID
        try:
            assistant_id = await get_assistant_id(assistant_type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid assistant type")

        logger.debug(f"Assistant ID: {assistant_id}")
        # Run the assistant and stream the response
        async with API_SEMAPHORE:
            run = await client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                stream=True
            )
            
            async for event in run:
                logger.debug(f"Received event: {type(event)}")
                if isinstance(event, ThreadMessageDelta):
                    delta = event.data.delta
                    if delta.content:
                        for content in delta.content:
                            if content.type == 'text':
                                yield f"data: {content.text.value}\n\n"
                elif isinstance(event, ThreadMessageCreated):
                    logger.info(f"Message created: {event}")
                elif isinstance(event, ThreadRunCompleted):
                    yield "data: [DONE]\n\n"
                    break
                else:
                    logger.warning(f"Unhandled event type: {type(event)}")

            # Fetch the final message after the run is completed
            messages = await client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=1)
            if messages.data:
                final_message = messages.data[0]
                logger.info(f"Final message: {final_message.id}")
                yield f"data: [FINAL]{final_message.id}\n\n"

    except Exception as e:
        logger.error(f"Error in process_chat: {str(e)}", exc_info=True)
        yield f"data: ERROR: {str(e)}\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
