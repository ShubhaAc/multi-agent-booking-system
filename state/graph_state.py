from typing import Optional
from pydantic import BaseModel


class GraphState(BaseModel):
    user_message: str = ""
    history: list[dict] = []
    intent: Optional[str] = None
    sender_id: Optional[str] = None
    doctor_name: Optional[str] = None
    recommended_doctors: list[str] = []
    appointment_date: Optional[str] = None
    appointment_time: Optional[str] = None
    duration_minutes: int = 30
    patient_name: Optional[str] = None
    phone_number: Optional[str] = None
    reason_for_visit: Optional[str] = None
    specialization_needed: Optional[str] = None
    invitee_email: Optional[str] = None
    appointment_id: Optional[int] = None
    suggested_alternative: Optional[str] = None
    rebook_requested: bool = False
    email_sent: bool = False
    cancellation_confirmed: bool = False
    response_message: str = ""
    vector_store: str = "chroma"

    class Config:
        extra = "forbid"