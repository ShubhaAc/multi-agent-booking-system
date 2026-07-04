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
async def send_invite_email(to_email: str, doctor_name: str, booking_date: str, booking_time: str, appointment_id: int) -> str:
    """Send a booking confirmation email to the patient."""
    msg = MIMEMultipart("alternative")
    msg["From"] = SMTP_SENDER
    msg["To"] = to_email
    msg["Subject"] = f"Confirmed: Dental Appointment with {doctor_name}"

    text_fallback = (
        f"Appointment Confirmation\n\n"
        f"Your dental appointment has been scheduled successfully.\n"
        f"Appointment ID: {appointment_id}\n"
        f"Practitioner: {doctor_name}\n"
        f"Date: {booking_date}\n"
        f"Time: {booking_time}\n\n"
        f"Please arrive 10 minutes prior to your time."
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f0f4f8; color: #334155; margin: 0; padding: 0; }}
            .email-container {{ max-width: 550px; margin: 40px auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }}
            .header {{ background-color: #0f766e; color: #ffffff; padding: 25px; text-align: center; }}
            .header h2 {{ margin: 0; font-size: 22px; font-weight: 500; letter-spacing: 0.5px; }}
            .content {{ padding: 30px; line-height: 1.6; }}
            .appointment-card {{ background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin: 20px 0; }}
            .info-row {{ margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 8px; }}
            .info-row:last-child {{ border-bottom: none; padding-bottom: 0; margin-bottom: 0; }}
            .label {{ font-weight: 600; color: #64748b; display: inline-block; width: 120px; }}
            .value {{ color: #0f172a; font-weight: 500; }}
            .footer {{ background-color: #f1f5f9; padding: 20px; text-align: center; font-size: 11px; color: #94a3b8; line-height: 1.5; border-top: 1px solid #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="header"><h2>Appointment Confirmed</h2></div>
            <div class="content">
                <p>Hello,</p>
                <p>Your upcoming dental visit has been successfully scheduled:</p>
                <div class="appointment-card">
                    <div class="info-row"><span class="label">Appointment ID:</span><span class="value">{appointment_id}</span></div>
                    <div class="info-row"><span class="label">Practitioner:</span><span class="value">{doctor_name}</span></div>
                    <div class="info-row"><span class="label">Date:</span><span class="value">{booking_date}</span></div>
                    <div class="info-row"><span class="label">Arrival Time:</span><span class="value" style="color:#0f766e;font-weight:bold;">{booking_time}</span></div>
                </div>
                <p style="font-size:13px;color:#64748b;"><em>Please arrive 10 minutes prior to your appointment.</em></p>
            </div>
            <div class="footer">BrightSmile Dental Clinic &copy; 2026<br>To modify your appointment, message our virtual coordinator.</div>
        </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(text_fallback, "plain"))
    msg.attach(MIMEText(html_content, "html"))
    try:
        await asyncio.to_thread(_send, msg, to_email)
        logger.info("Invite sent to %s", to_email)
        return f"Invite email successfully sent to {to_email}."
    except Exception as e:
        logger.error("Failed to send invite email: %s", e)
        return f"Failed to send email to {to_email}: {str(e)}"


@tool
async def send_cancellation_email(to_email: str, doctor_name: str, booking_date: str, booking_time: str, appointment_id: int) -> str:
    """Send a cancellation confirmation email to the patient."""
    msg = MIMEMultipart("alternative")
    msg["From"] = SMTP_SENDER
    msg["To"] = to_email
    msg["Subject"] = f"Cancelled: Dental Appointment with {doctor_name}"

    text_fallback = (
        f"Appointment Cancellation\n\n"
        f"Your dental appointment has been cancelled.\n"
        f"Appointment ID: {appointment_id}\n"
        f"Practitioner: {doctor_name}\n"
        f"Date: {booking_date}\n"
        f"Time: {booking_time}\n\n"
        f"If this was a mistake, or you'd like to rebook, just let us know."
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f0f4f8; color: #334155; margin: 0; padding: 0; }}
            .email-container {{ max-width: 550px; margin: 40px auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }}
            .header {{ background-color: #b91c1c; color: #ffffff; padding: 25px; text-align: center; }}
            .header h2 {{ margin: 0; font-size: 22px; font-weight: 500; }}
            .content {{ padding: 30px; line-height: 1.6; }}
            .appointment-card {{ background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin: 20px 0; }}
            .info-row {{ margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 8px; }}
            .info-row:last-child {{ border-bottom: none; padding-bottom: 0; margin-bottom: 0; }}
            .label {{ font-weight: 600; color: #64748b; display: inline-block; width: 120px; }}
            .value {{ color: #0f172a; font-weight: 500; }}
            .footer {{ background-color: #f1f5f9; padding: 20px; text-align: center; font-size: 11px; color: #94a3b8; border-top: 1px solid #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="header"><h2>Appointment Cancelled</h2></div>
            <div class="content">
                <p>Hello,</p>
                <p>The following appointment has been cancelled:</p>
                <div class="appointment-card">
                    <div class="info-row"><span class="label">Appointment ID:</span><span class="value">{appointment_id}</span></div>
                    <div class="info-row"><span class="label">Practitioner:</span><span class="value">{doctor_name}</span></div>
                    <div class="info-row"><span class="label">Date:</span><span class="value">{booking_date}</span></div>
                    <div class="info-row"><span class="label">Time:</span><span class="value">{booking_time}</span></div>
                </div>
                <p style="font-size:13px;color:#64748b;">If you'd like to rebook, just message us anytime.</p>
            </div>
            <div class="footer">BrightSmile Dental Clinic &copy; 2026</div>
        </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(text_fallback, "plain"))
    msg.attach(MIMEText(html_content, "html"))
    try:
        await asyncio.to_thread(_send, msg, to_email)
        logger.info("Cancellation email sent to %s", to_email)
        return f"Cancellation email successfully sent to {to_email}."
    except Exception as e:
        logger.error("Failed to send cancellation email: %s", e)
        return f"Failed to send cancellation email to {to_email}: {str(e)}"


@tool
async def send_reschedule_email(to_email: str, doctor_name: str, new_date: str, new_time: str, appointment_id: int) -> str:
    """Send a reschedule confirmation email to the patient."""
    msg = MIMEMultipart("alternative")
    msg["From"] = SMTP_SENDER
    msg["To"] = to_email
    msg["Subject"] = f"Updated: Dental Appointment with {doctor_name}"

    text_fallback = (
        f"Appointment Rescheduled\n\n"
        f"Your dental appointment has been updated.\n"
        f"Appointment ID: {appointment_id}\n"
        f"Practitioner: {doctor_name}\n"
        f"New Date: {new_date}\n"
        f"New Time: {new_time}\n\n"
        f"Please arrive 10 minutes prior to your time."
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f0f4f8; color: #334155; margin: 0; padding: 0; }}
            .email-container {{ max-width: 550px; margin: 40px auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }}
            .header {{ background-color: #1d4ed8; color: #ffffff; padding: 25px; text-align: center; }}
            .header h2 {{ margin: 0; font-size: 22px; font-weight: 500; letter-spacing: 0.5px; }}
            .content {{ padding: 30px; line-height: 1.6; }}
            .appointment-card {{ background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin: 20px 0; }}
            .info-row {{ margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 8px; }}
            .info-row:last-child {{ border-bottom: none; padding-bottom: 0; margin-bottom: 0; }}
            .label {{ font-weight: 600; color: #64748b; display: inline-block; width: 120px; }}
            .value {{ color: #0f172a; font-weight: 500; }}
            .footer {{ background-color: #f1f5f9; padding: 20px; text-align: center; font-size: 11px; color: #94a3b8; line-height: 1.5; border-top: 1px solid #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="header"><h2>Appointment Rescheduled</h2></div>
            <div class="content">
                <p>Hello,</p>
                <p>Your dental appointment has been updated to a new date and time:</p>
                <div class="appointment-card">
                    <div class="info-row"><span class="label">Appointment ID:</span><span class="value">{appointment_id}</span></div>
                    <div class="info-row"><span class="label">Practitioner:</span><span class="value">{doctor_name}</span></div>
                    <div class="info-row"><span class="label">New Date:</span><span class="value">{new_date}</span></div>
                    <div class="info-row"><span class="label">New Time:</span><span class="value" style="color:#1d4ed8;font-weight:bold;">{new_time}</span></div>
                </div>
                <p style="font-size:13px;color:#64748b;"><em>Please arrive 10 minutes prior to your appointment.</em></p>
            </div>
            <div class="footer">BrightSmile Dental Clinic &copy; 2026<br>To modify again, message our virtual coordinator.</div>
        </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(text_fallback, "plain"))
    msg.attach(MIMEText(html_content, "html"))
    try:
        await asyncio.to_thread(_send, msg, to_email)
        logger.info("Reschedule email sent to %s", to_email)
        return f"Reschedule confirmation email successfully sent to {to_email}."
    except Exception as e:
        logger.error("Failed to send reschedule email: %s", e)
        return f"Failed to send reschedule email to {to_email}: {str(e)}"