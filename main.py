import asyncio
import logging
from dotenv import load_dotenv
from db.schema import init_db
from graph import build_graph
from state import GraphState
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

async def main():
    await init_db()
    graph = build_graph()
    logger.info("Booking system ready.")

    while True:
      user_input = input("\nYou: ").strip()
      if user_input.lower() in ("exit", "quit"):
          print("Goodbye!")
          break
      
      initial_state = GraphState(user_message=user_input, booked_by="user@company.com")
      result = await graph.ainvoke(initial_state)
      print(f"\nBot: {result['response_message']}")

if __name__ == "__main__":
    asyncio.run(main())