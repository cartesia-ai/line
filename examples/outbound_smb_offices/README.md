# SMB Business Data Collection

This is an example of a voice agent that outbounds small business to ask a series of questions from them.

This is a single prompt agent that handles direct pickup, IVR and voicemail. If it reaches a human, it will ask a simple questionaire. In this case, it will try to confirm the name and address.

## Problem

When you call a small business, one of three things will happen:

| Outcome       | Description                            | Expected Agent Behavior |
| ------------- | -------------------------------------- | ----------------------  |
| Direct Pickup | someone at the front desk will pick up | Begin the questionaire  |
| IVR           | the business has a phone tree          | Select the best option  |
| Voicemail     | we're calling when they are closed     | end the call            |

