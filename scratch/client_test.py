import requests
import time

def test_stream():
    url = "http://127.0.0.1:8002/stream"
    payload = {
        "text": "Hello, this is a test of the emergency calling system streaming capability. We are generating audio sentence by sentence to ensure the lowest possible latency for the caller.",
        "voice": "male, american accent"
    }

    print(f"Connecting to {url}...")
    start_time = time.time()
    
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        print("Stream connected!")
        
        chunk_count = 0
        total_bytes = 0
        
        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                if chunk_count == 0:
                    first_byte_time = time.time() - start_time
                    print(f"Time to first chunk: {first_byte_time:.2f}s")
                
                chunk_count += 1
                total_bytes += len(chunk)
                print(f"Received chunk {chunk_count}: {len(chunk)} bytes")
        
        total_time = time.time() - start_time
        print(f"\nStream finished.")
        print(f"Total chunks: {chunk_count}")
        print(f"Total bytes: {total_bytes}")
        print(f"Total time: {total_time:.2f}s")

if __name__ == "__main__":
    # Wait a bit for server to start if run sequentially
    test_stream()
