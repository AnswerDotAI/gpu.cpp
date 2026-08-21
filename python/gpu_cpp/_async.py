import asyncio


async def wait(future):
    while not future._poll(): await asyncio.sleep(0.001)
    return future._result()
