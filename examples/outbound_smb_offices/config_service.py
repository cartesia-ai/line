"""
Service for dynamically changing the provider per phone number

# Runs service on port 8001
uv run config_service.py

# Test
curl -iX POST http://localhost:8001/provider \
    -H "Content-Type: application/json" \
    -d '{"to": "+17173030505"}'
"""

from collections.abc import Generator

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
import uvicorn

CONFIG = """
Jane Doe | 143 Main St, Anytown | 1234567890
John Smith | 111 West 1st St, Whicheverville | 1234567890
"""


class BusinessDetails(BaseModel):
    name: str
    address: str
    phone_number: str


def to_E164(phone_number: str) -> str:
    """
    Converts a phone number to E164 format.
    """
    return f"+1{phone_number.replace('(', '').replace(')', '').replace('-', '').replace(' ', '')}"


def parse_config(config: str) -> Generator[BusinessDetails, None, None]:
    """
    Parses the config and returns a list of business details.
    """
    for line in config.split("\n"):
        if line.strip():
            name, address, phone_number = line.split("|")
            yield BusinessDetails(
                name=name.strip(),
                address=address.strip(),
                phone_number=to_E164(phone_number.strip()),
            )


CONFIG_DICT = {x.phone_number: x for x in parse_config(CONFIG)}

app = FastAPI()


class CallRequestPartial(BaseModel):
    to: str


@app.post("/details")
async def get_details(request: CallRequestPartial) -> BusinessDetails:
    """
    Returns a hardcoded provider based on the call request.
    """
    details = CONFIG_DICT.get(request.to, None)

    if not details:
        logger.warning(f"Provider for phone number not found. {request.to}")

        return BusinessDetails(
            name="Jane Doe",
            address="143 Main St, Anytown",
            phone_number="+14086121089",
        )

        raise HTTPException(
            status_code=404,
            detail=f"Provider for phone number not found. {request.to}",
        )

    return details


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
