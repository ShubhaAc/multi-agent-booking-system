import aiosqlite
import logging
from config import DB_PATH

logger = logging.getLogger(__name__)

async def check_availability(room_name: str, booking_date: str, booking_time: str, duration_minutes: int = 60) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("""
            SELECT b.id FROM bookings b
            JOIN rooms r ON b.room_id = r.id
            WHERE r.name = ?
            AND b.booking_date = ?
            AND b.status = 'active'
            AND (
                (substr(b.booking_time, 1, 2) * 60 + substr(b.booking_time, 4, 2))
                    < (substr(?, 1, 2) * 60 + substr(?, 4, 2)) + ?
                AND
                (substr(b.booking_time, 1, 2) * 60 + substr(b.booking_time, 4, 2))
                    + b.duration_minutes
                    > (substr(?, 1, 2) * 60 + substr(?, 4, 2))
            )
        """, (room_name, booking_date, booking_time, booking_time, duration_minutes, booking_time, booking_time)) as cursor:
            return await cursor.fetchone() is None
        

async def create_booking(room_name: str, booked_by: str, invitee_email: str, booking_date: str, booking_time: str, duration_minutes: int = 60) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT id FROM rooms WHERE name = ?", (room_name,)) as cursor:
            room = await cursor.fetchone()
        result = await db.execute("""
            INSERT INTO bookings (room_id, booked_by, invitee_email, booking_date, booking_time, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (room[0], booked_by, invitee_email, booking_date, booking_time, duration_minutes))
        await db.commit()
        logger.info("Booking created with id %s", result.lastrowid)
        return result.lastrowid

async def cancel_booking(booking_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE bookings SET status = 'cancelled' WHERE id = ? AND status = 'active'
        """, (booking_id,))
        await db.commit()
        logger.info("Booking %s cancelled.", booking_id)
        return True
    
async def find_alternative_room(exclude_room: str, booking_date: str, booking_time: str, duration_minutes: int = 60) -> str | None:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("""
            SELECT r.name FROM rooms r
            WHERE r.name != ? AND r.id NOT IN (
                SELECT b.room_id FROM bookings b
                WHERE b.booking_date = ?
                AND b.status = 'active'
                AND (
                    (substr(b.booking_time, 1, 2) * 60 + substr(b.booking_time, 4, 2))
                        < (substr(?, 1, 2) * 60 + substr(?, 4, 2)) + ?
                    AND
                    (substr(b.booking_time, 1, 2) * 60 + substr(b.booking_time, 4, 2))
                        + b.duration_minutes
                        > (substr(?, 1, 2) * 60 + substr(?, 4, 2))
                )
            )
            LIMIT 1
        """, (exclude_room, booking_date, booking_time, booking_time, duration_minutes, booking_time, booking_time)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None
        


async def get_all_rooms() -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT name FROM rooms") as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]