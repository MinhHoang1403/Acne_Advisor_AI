import asyncio
import os
from dotenv import load_dotenv

# Load env before importing anything else
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

import redis.asyncio as aioredis

async def clear_cache():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    print(f"Connecting to {redis_url}...")
    try:
        redis = aioredis.from_url(redis_url, decode_responses=True)
        await redis.ping()
    except Exception as e:
        print(f"Redis is not available: {e}")
        return
        
    print("Clearing cache keys: cache:answer:*")

    try:
        # Use scan to find keys and delete them
        cursor = 0
        deleted_count = 0
        while True:
            cursor, keys = await redis.scan(cursor, match="cache:answer:*", count=100)
            if keys:
                await redis.delete(*keys)
                deleted_count += len(keys)
            if cursor == 0:
                break

        print(f"Deleted {deleted_count} cache keys.")
    finally:
        await redis.aclose()

if __name__ == "__main__":
    asyncio.run(clear_cache())
