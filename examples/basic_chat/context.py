from datetime import datetime, timedelta, time, date
import zoneinfo
from typing import Optional, List, Dict, Literal, Union
from pydantic import BaseModel, Field

class WebhookChunkItem(BaseModel):
    start_timestamp: Optional[float] = None
    text: Optional[str] = None

class WebhookBodyItem(BaseModel):
    end_reason: Optional[str] = None
    end_timestamp: Optional[float] = None
    role: Optional[str] = None
    start_timestamp: Optional[float] = None
    text: Optional[str] = None
    text_chunks: Optional[List[WebhookChunkItem]] = None
    tts_ttfb: Optional[float] = None

class CartesiaWebhookPayload(BaseModel):
    type: Optional[str] = None
    request_id: Optional[str] = None
    agent_id: Optional[str] = None
    webhook_id: Optional[str] = None
    call_id: Optional[str] = None
    body: Optional[Union[List[WebhookBodyItem], WebhookBodyItem]] = None
    timestamp: Optional[str] = None

class Driver(BaseModel):
    name: Optional[str] = None          # Full name
    dob: Optional[str] = None           # YYYY-MM-DD
    license_number: Optional[str] = None

class Scheduling(BaseModel):
    when: Optional[str] = None
    notes: Optional[str] = None


class Vehicle(BaseModel):
    model: Optional[str] = None
    notes: Optional[str] = None


class AutoPolicyData(BaseModel):
    annual_premium: str | None = None
    vehicles: list[Vehicle] = []
    main_customer: Driver
    additional_drivers: list[Driver] = []


# Note: the order of fields matters here
# LLMs are most accurate when they use chain-of-thought: evidence, then reasoning, then final result
# therefore we ask first for various fields that are also evidence for the final outcome
# then for reasoning (qualification_reasoning)
# and finally for result (qualified and dq_code)
class ExtractedData(BaseModel):
    confirmed_property_address: str | None = None
    confirmed_dob: date | None = None
    transaction_type: Optional[Literal["purchase", "refinance"]] = None
    property_type: Optional[Literal["primary", "secondary", "landlord"]] = None
    auto_policy_data: AutoPolicyData | None = None
    additional_bundling_opportunities: str | None = None
    callback_scheduled: str = Field(
        default="",
        description="Date and time of the scheduled callback in MM/dd/yyyy HH:mm:ss format in customer's timezone",
        examples=["12/25/2024 14:30:00"],
    )
    time_zone: str = Field(
        default="",
        description="The time zone of the customer's location",
        examples=["America/New_York"],
    )
    qualification_reasoning: str
    qualified: Literal["yes", "no", "pending"]
    dq_code: Optional[Literal["bad-contact-info", "declined", "do-not-contact", "duplicate"]]



class TimeZoneInfo(BaseModel):
    timezone_IANA: str = Field(description="IANA timezone name")


def round_up(dt: datetime) -> datetime:
    # round up to the next 30 minute interval
    # no need to worry about transition to next day for this application
    if dt.minute < 30:
        return dt.replace(minute=30)
    else:
        return dt.replace(minute=0, hour=dt.hour+1)

NON_WORKING_DAYS = [5, 6]  # Sat, Sun

TIME_FORMAT = "%A, %B %d, %Y at %I:%M %p %Z"

def format_availability(dt1: datetime, dt2: datetime) -> str:
    return f'between {dt1.strftime(TIME_FORMAT)} and {dt2.strftime(TIME_FORMAT)} (customer time)'

def find_availability(
    tzinfo: TimeZoneInfo
) -> str:
    customer_tz = zoneinfo.ZoneInfo(tzinfo.timezone_IANA)
    az_tz = zoneinfo.ZoneInfo('America/Phoenix')
    now_az = datetime.now(az_tz)
    cob_time = time(hour=17, minute=0)
    cob = datetime.combine(now_az.date(), cob_time, tzinfo=az_tz)
    if cob - now_az > timedelta(hours=3, minutes=30) and now_az.date().weekday() not in NON_WORKING_DAYS:
        today_start = round_up(now_az + timedelta(hours=3)).astimezone(customer_tz)
        today_end = cob.astimezone(customer_tz)
        todays_availability = format_availability(today_start, today_end)
    else:
        todays_availability = 'none'

    next_day = now_az.date()
    # TODO: take holidays into account too
    while (next_day := next_day + timedelta(days=1)).weekday() in NON_WORKING_DAYS:
        pass
    next_day_start = datetime.combine(next_day, time(hour=6, minute=0), tzinfo=az_tz).astimezone(customer_tz)
    next_day_end = datetime.combine(next_day, time(hour=17, minute=0), tzinfo=az_tz).astimezone(customer_tz)
    next_day_availability = format_availability(next_day_start, next_day_end)

    return f"""
    Current customer time: {now_az.astimezone(customer_tz).strftime(TIME_FORMAT)}
    Today's availability: {todays_availability}
    Next working day availability: {next_day_availability}
    General availability: Mon-Fri, 6 AM - 5 PM US/Arizona time
    """
