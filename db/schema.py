import aiosqlite
import logging
from config import DB_PATH

logger = logging.getLogger(__name__)

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                specialization TEXT,
                available_days TEXT,
                start_time TEXT,
                end_time TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id INTEGER NOT NULL,
                patient_email TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                phone_number TEXT,
                reason_for_visit TEXT,
                sender_id TEXT,
                appointment_date TEXT NOT NULL,
                appointment_time TEXT NOT NULL,
                duration_minutes INTEGER DEFAULT 30,
                status TEXT DEFAULT 'active',
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            )
        """)
        async with db.execute("SELECT COUNT(*) FROM doctors") as cursor:
            count = (await cursor.fetchone())[0]
        if count == 0:
            await _seed_doctors(db)
        await db.commit()
        logger.info("Database initialised.")

async def _seed_doctors(db):
    doctors = [
        ("Dr. Ananya Sharma",    "General & Preventive Dentistry", "Monday,Wednesday,Friday",                           "09:00", "17:00"),
        ("Dr. Rohan Mehta",      "Endodontics",                    "Tuesday,Thursday,Saturday",                         "10:00", "18:00"),
        ("Dr. Priya Nair",       "Orthodontics",                   "Monday,Tuesday,Friday",                             "09:00", "16:00"),
        ("Dr. Samuel Okafor",    "Oral Surgery",                   "Wednesday,Thursday,Friday",                         "08:00", "15:00"),
        ("Dr. Liu Wei",          "Cosmetic Dentistry",             "Monday,Wednesday,Saturday",                         "10:00", "18:00"),
        ("Dr. Fatima Al-Hassan", "Paediatric Dentistry",           "Tuesday,Thursday,Saturday",                         "09:00", "17:00"),
        ("Dr. Carlos Rivera",    "Periodontics",                   "Monday,Thursday,Friday",                            "08:00", "16:00"),
        ("Dr. Mei Tanaka",       "Prosthodontics",                 "Tuesday,Wednesday,Saturday",                        "09:00", "17:00"),
        ("Dr. James O'Brien",    "Emergency Dentistry",            "Monday,Tuesday,Wednesday,Thursday,Friday,Saturday", "08:00", "20:00"),
        ("Dr. Aisha Patel",      "General Dentistry",              "Wednesday,Thursday,Saturday",                       "10:00", "18:00"),
        ("Dr. Nikolai Volkov",   "Dental Radiology",               "Monday,Friday",                                     "09:00", "15:00"),
        ("Dr. Sara Kim",         "Implantology",                   "Tuesday,Thursday",                                  "09:00", "17:00"),
    ]
    await db.executemany(
        "INSERT INTO doctors (name, specialization, available_days, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
        doctors
    )
    logger.info("Seeded %d doctors.", len(doctors))