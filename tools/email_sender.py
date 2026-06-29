import asyncio
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.tools import tool
from config import SMTP_HOST, SMTP_PORT, SMTP_PASSWORD, SMTP_SENDER

logger = logging.getLogger(__name__)

def _send(msg: MIMEMultipart, to_email: str):
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_SENDER, SMTP_PASSWORD)
        server.sendmail(SMTP_SENDER, to_email, msg.as_string())

@tool
async def send_invite_email(to_email: str, room_name: str, booking_date: str, booking_time: str) -> str:
    """Send a meeting invite email to the invitee after a successful booking."""
    msg = MIMEMultipart()
    msg["From"] = SMTP_SENDER
    msg["To"] = to_email
    msg["Subject"] = f"Meeting Invite: {room_name}"
    msg.attach(MIMEText(f"You have been invited to a meeting in {room_name} on {booking_date} at {booking_time}.", "plain"))
    try:
        await asyncio.to_thread(_send, msg, to_email)
        logger.info("Invite sent to %s", to_email)
        return f"Invite email successfully sent to {to_email}."
    except Exception as e:
        logger.error("Failed to send email: %s", e)
        return f"Failed to send email to {to_email}: {str(e)}"