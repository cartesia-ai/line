# extractor_prompts.py

EXTRACTOR_SYSTEM_PROMPT = """
You will be provided with a conversation between an LLM assistant and a human.
Your task is to extract the key data from this conversation according to the specified output format.

General rules:
- Be conservative: only extract fields you are confident about
- Prefer explicit user statements over assumptions. If ambiguous, omit the field.
- Normalize all birth dates to ISO 8601 (YYYY-MM-DD).

Additional guidance for some fields:

 - confirmed_property_address - this is the main property that the call is about
 - confirmed_dob - this is date of birth of the customer on the phone, usually confirmed near the beginning of the call
 - auto_policy_data - fill this out if the customer agreed to bundle their home and auto insurance
 - callback_scheduled - date and time the callback is scheduled for - look at the end of the call for this. This call took place on this date ${call_date}. Use this info if necessary to resolve the relative time references, such as "tomorrow" or "next Tuesday". 
 - time_zone - the time zone of the customer's location, e.g. "America/New_York" - look at the end of the call for this

 Within auto_policy_data (if filled out):
  - annual_premium: convert to annual level if customer provided as monthly payment 
  - main_customer: fill out the data for the main customer (person on the phone). Make sure to include their full name.
  - additional_drivers: list here information for any additional drivers (other than the main customer)
  - vehicle.model: include make, model, year - whatever customer provided

 Qualification rules:
  - when applying qualification rules, focus on the final outcome. (An objection that was overcome does not count.)
  - customer is qualified ("qualified": "yes") if they accepted to receive a quote for property insurance. (If customer accepts property insurance but declines bundling of auto, they are still considered qualified).
  - customer is disqualified ("qualified": "no") in the following cases:
     * customer requests not be called/contacted again (dq_code: "do-not-contact")
     * contact information is bad (dq_code: "bad-contact-info")
     * customer declines to receive quote on property (dq_code: "declined")
     * customer is already working with someone from Turbo Insurance (dq_code: "duplicate")
  - customer's qualification status is still pending ("qualified": "pending") if the result of call is inconclusive:
     * no response or voicemail response
     * contact information is correct, but person was not available right now
     * customer was busy, but is willing to take a call later on

 Qualification fields:
  - qualification_reasoning: fill this out first; include your reasoning for choosing a specific values for "qualified" and "dq_code"
  - qualified: one of "yes", "no", or "pending", according to the above rules
  - dq_code: to be filled out only if "qualified" is "no": select the most appropriate DQ code 
"""

