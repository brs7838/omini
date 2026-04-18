import asyncio
import websockets
import json

async def test_connect():
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            # Send small dummy audio bytes
            await websocket.send(b"\x00" * 4096)
            print("Sent audio")
            # Wait for response
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=5)
                print("Received:", msg)
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connect())
