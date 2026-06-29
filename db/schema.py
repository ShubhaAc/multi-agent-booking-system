"""
Two connected tables:

1. rooms: id (PK) 
   name (Unique: Matrix, Pixel, Nebula, Echo)
2. bookings: id (PK) 
   room_id (FK -> rooms.id) | booked_by | invitee_email | booking_date | booking_time | status ('active'/'cancelled')
"""
import aiosqlite
import asyncio
import logging

logger = logging.getLogger(__name__)

DB_PATH = "db/bookings.db"

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id INTEGER NOT NULL,
                booked_by TEXT NOT NULL,
                invitee_email TEXT,
                booking_date TEXT NOT NULL,
                booking_time TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL DEFAULT 60,
                status TEXT NOT NULL DEFAULT 'active',
                FOREIGN KEY (room_id) REFERENCES rooms(id)
            )
        """)
        await db.executemany(
            "INSERT OR IGNORE INTO rooms (name) VALUES (?)",
            [("Matrix Room",), ("Pixel Room",), ("Nebula Room",), ("Echo Room",)]
        )
        await db.commit()
        logger.info("Database initialised successfully.")

if __name__ == "__main__":
    asyncio.run(init_db())