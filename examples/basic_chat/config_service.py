"""
Service for dynamically changing the provider per phone number

# Runs service on port 8001
uv run config_service.py

# Test
curl -iX POST http://localhost:8001/provider \
    -H "Content-Type: application/json" \
    -d '{"to": "+155573030505"}'
"""
import logging
import string
from collections.abc import Generator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from loguru import logger
from pydantic import BaseModel
import uvicorn


from typing import List, Literal, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict

from context import ExtractedData, CartesiaWebhookPayload
import os
import openai

from datetime import datetime
import zoneinfo

from extractor_prompt import EXTRACTOR_SYSTEM_PROMPT

import json
from datetime import date
import sys

# Remove default handler (optional but recommended)
logger.remove()

# Add handler: standard output
logger.add(
    sink=sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

# Add handler: file output
logger.add(
    "app.log",
    #rotation="10 MB",   # optional: rotate logs automatically
    #retention="10 days",  # optional: keep logs for 10 days
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai_client = openai.AsyncClient(api_key=os.getenv("OPENAI_API_KEY"))
else:
    openai_client = None


class BodyItem(BaseModel):
    end_timestamp: float
    role: str
    start_timestamp: float
    text: str
    tts_ttfb: Optional[float] = None

    model_config = ConfigDict(extra="allow")  # allow extra attributes inside body items


class WebhookEvent(BaseModel):
    type: str
    request_id: str
    agent_id: str
    webhook_id: str
    body: List[BodyItem]
    timestamp: str

    model_config = ConfigDict(extra="allow")  # allow extra attributes at top level



class BusinessDetails(BaseModel):
    customer_name: str
    loan_officer_name: str
    lender_name: str
    mortgage_office_name: str
    phone_number: str
    property_address: str
    dob_on_file: str

def serialize(obj):
    """Custom serializer for non-serializable objects."""
    if isinstance(obj, date):
        return obj.isoformat()
    # fallback for other non-serializable objects
    return str(obj)

def to_E164(phone_number: str) -> str:
    """
    Converts a phone number to E164 format.
    """
    phone_number = (
        phone_number
        .lstrip('+')
        .replace('(', '')
        .replace(')', '')
        .replace('-', '')
        .replace(' ', '')
        .replace('.', '')
    )
    if len(phone_number) == 10:
        return f'+1{phone_number}'
    elif len(phone_number) == 11:
        return f'+{phone_number}'
    raise ValueError(phone_number)


def convert_to_arizona(callback_scheduled: str, time_zone: str) -> str:
    """
    Converts a datetime string from a given timezone to Arizona time.

    Args:
        callback_scheduled (str): datetime string in "MM/DD/YYYY HH:MM:SS" format
        time_zone (str): IANA timezone string, e.g., "America/Chicago"

    Returns:
        str: datetime string in Arizona time, same format
    """
    # Parse the input datetime string
    input_dt = datetime.strptime(callback_scheduled, "%m/%d/%Y %H:%M:%S")

    # Apply the input timezone
    customer_tz = zoneinfo.ZoneInfo(time_zone)
    input_dt = input_dt.replace(tzinfo=customer_tz)

    # Convert to Arizona time
    az_tz = zoneinfo.ZoneInfo("America/Phoenix")
    az_dt = input_dt.astimezone(az_tz)

    # Return formatted string
    return az_dt.strftime("%m/%d/%Y %H:%M:%S")

def parse_config() -> Generator[BusinessDetails, None, None]:
    """
    Parses the config and returns a list of business details.
    """

    yield BusinessDetails(
            customer_name="John Smith",
            loan_officer_name="Cashmore Greed",
            lender_name="Shady Oaks Lending",
            mortgage_office_name="Chicago Branch",
            phone_number=to_E164('666.666.6666'),
            property_address="123 Sesame Street, Apt 13, Chicago, IL 60007",
            dob_on_file="01/01/1970",
        )

    yield BusinessDetails(
        customer_name="Marc Bernstein",
        loan_officer_name="Cashmore Greed",
        lender_name="Shady Oaks Lending",
        mortgage_office_name="Chicago Branch",
        phone_number=to_E164('555.555.5555'),
        property_address="123 Sesame Street, Apt 13, Chicago, IL 60007",
        dob_on_file="10/26/1985",
    )



CONFIG_DICT = {x.phone_number: x for x in parse_config()}

app = FastAPI()


class CallRequestPartial(BaseModel):
    to: str
    from_: str | None = Field(alias="from", default=None)

async def extract_structured_data_from_webhook(
    event: CartesiaWebhookPayload,
) -> ExtractedData:

    # 1) prepare transcript
    messages: List[Dict[str, Any]] = [
        {"role": item.role, "content": item.text}
        for item in event.body
        if item.role in ("user", "assistant") and item.text
    ]

    # fallback if messages do not exist
    if not messages:
        logger.warning(
            f"No user/assistant messages found in webhook body for request_id={event.request_id}"
        )
        pass

    logger.info(f'call transcript: {messages}')

    az_tz = zoneinfo.ZoneInfo('America/Phoenix')
    now_az = datetime.now(az_tz)
    call_date = now_az.date()

    prompt = string.Template(EXTRACTOR_SYSTEM_PROMPT).substitute(call_date=call_date)


    # 2) call LLM
    response = await openai_client.responses.parse(
        model="gpt-5-mini",
        instructions=prompt,
        input=messages,
        text_format=ExtractedData,
        reasoning={"effort": "high"},
    )

    parsed: ExtractedData = response.output_parsed

    # 3) Convert to JSON
    parsed_json = json.dumps(parsed.model_dump(), default=serialize, indent=2)
    logger.info(f"Parsed extracted data (JSON) for request_id={event.request_id}:\n{parsed_json}")
    #logger.info(convert_to_arizona(callback_scheduled=parsed.callback_scheduled, time_zone=parsed.time_zone))

    return parsed


@app.post("/details")
async def get_details(request: CallRequestPartial) -> BusinessDetails:
    """
    Returns a hardcoded provider based on the call request.
    """
    try:
        phone_number_to = to_E164(request.to)
        phone_number_from = to_E164(request.from_) if request.from_ else None
    except ValueError:
        logger.info(f'Could not normalize phone numbers {request}, picking default entry')
        return next(iter(CONFIG_DICT.values()))

    if phone_number_to in CONFIG_DICT:
        return CONFIG_DICT[phone_number_to]
    elif phone_number_from in CONFIG_DICT:
        return CONFIG_DICT[phone_number_from]
    else:
        logger.info(f'PHONE NUMBERS {request} NOT FOUND, PICKING FIRST ENTRY')
        return next(iter(CONFIG_DICT.values()))


@app.post("/cartesia-webhook")
async def cartesia_webhook(
        payload: CartesiaWebhookPayload,
        x_webhook_secret: Optional[str] = Header(default=None, convert_underscores=False, alias="x-webhook-secret"),
):
    logging.info("finishing cartesia logging")
    logging.info(f"PAYLOAD TYPE: {payload.type}")
    expected_secret = os.getenv("CARTESIA_WEBHOOK_SECRET")
    if not expected_secret:
        raise HTTPException(
            status_code=500,
            detail="CARTESIA_WEBHOOK_SECRET not configured on server.",
        )

    if x_webhook_secret != expected_secret:
       raise HTTPException(status_code=401, detail="Invalid webhook secret.")



    if payload.type == "call_completed" or payload.type == "call_failed":
        transcript_lines = []
        for item in payload.body:
            if item.text and item.role:
                line = f"{item.role}: {item.text}"
                transcript_lines.append(line)
        #save_transcript(payload)
        #call LLM
        llm_response = await extract_structured_data_from_webhook(event=payload)
        #logging.info(f"Extracted transcript: {llm_response}")
        pass
    elif payload.type == "call_failed":
        # call LLM
        pass

    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run('config_service:app', host="0.0.0.0", port=8001, reload=True)
