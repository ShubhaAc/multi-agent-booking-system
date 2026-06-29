from typing import Optional
from pydantic import BaseModel

class GraphState(BaseModel):
    
    """The shared notebook that the chatbot agents pass around.
    
    How it works:
    1. The user types a message -> saved in `user_message`.
    2. The AI reads it, figures out what to do -> saved in `intent`.
    3. The agents collect room details, check the database, and send emails.
    4. The final reply is saved in `response_message` and shown to the user.
    """
  
    user_message: str = ""    # user prompt
    history: list[dict] = []
    intent: Optional[str] = None  # intent : booking/ canceling/ rescheduling
    room_name: Optional[str] = None
    booking_date: Optional[str] = None
    booking_time: Optional[str] = None
    duration_minutes: int = 60
    booked_by: Optional[str] = None
    invitee_email: Optional[str] = None
    booking_id: Optional[int] = None
    suggested_alternative: Optional[str] = None
    email_sent: bool = False
    cancellation_confirmed: bool = False
    response_message: str = ""   #final response

    class Config:

      """A safety guardrail that blocks any unknown or unmapped fields 
        from being accidentally added to the shared state memory.
      """

      extra = "forbid"   # strict schema guardrail