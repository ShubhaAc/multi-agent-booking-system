import sqlite3
from config import DB_PATH

con = sqlite3.connect(DB_PATH)
rows = con.execute("""
    SELECT a.id, a.appointment_date, a.appointment_time, a.duration_minutes, 
           a.status, d.name, a.patient_name, a.patient_email, 
           a.phone_number, a.reason_for_visit
    FROM appointments a
    JOIN doctors d ON a.doctor_id = d.id
    ORDER BY a.id
""").fetchall()
for row in rows:
    print(row)