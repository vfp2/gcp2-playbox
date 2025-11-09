#!/usr/bin/env python3
"""
Test if GCP2.NET device validates SSL certificates

This creates a local MQTT proxy with a self-signed certificate
to see if the device will connect (indicating cert validation is disabled)
"""
import socket
import ssl
import threading
from datetime import datetime

LISTEN_PORT = 8883
TARGET_HOST = 'a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com'
TARGET_PORT = 8883

def create_self_signed_cert():
    """Generate a self-signed certificate for testing"""
    import subprocess

    print("Generating self-signed certificate...")
    result = subprocess.run([
        'openssl', 'req', '-new', '-x509', '-days', '1',
        '-nodes', '-out', 'test_proxy.crt', '-keyout', 'test_proxy.key',
        '-subj', f'/CN={TARGET_HOST}'
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ Certificate generated: test_proxy.crt / test_proxy.key")
        return True
    else:
        print(f"✗ Certificate generation failed: {result.stderr}")
        return False

def handle_connection(client_ssl, addr):
    """Handle a connection attempt"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] ✓ CLIENT CONNECTED from {addr}")
    print(f"[{timestamp}] → Device accepted self-signed certificate!")
    print(f"[{timestamp}] → Certificate validation appears to be DISABLED")

    try:
        # Try to read some data
        data = client_ssl.recv(1024)
        if data:
            print(f"[{timestamp}] → Received {len(data)} bytes")
            print(f"[{timestamp}] → First packet: {data[:32].hex()}")

            # Could forward to real server here
            print("\nConclusion: You can run a local MITM proxy!")
            print("The device does NOT strictly validate AWS certificates.")
    except Exception as e:
        print(f"Error reading data: {e}")
    finally:
        client_ssl.close()

def main():
    # Generate cert if needed
    import os
    if not os.path.exists('test_proxy.crt') or not os.path.exists('test_proxy.key'):
        if not create_self_signed_cert():
            return

    # Create SSL context with self-signed cert
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain('test_proxy.crt', 'test_proxy.key')
    except Exception as e:
        print(f"Error loading certificate: {e}")
        return

    # Create listening socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', LISTEN_PORT))
    server.listen(1)

    print("\n" + "="*60)
    print("CERTIFICATE VALIDATION TEST")
    print("="*60)
    print(f"\nListening on port {LISTEN_PORT} with self-signed certificate")
    print(f"Certificate CN: {TARGET_HOST}\n")
    print("INSTRUCTIONS:")
    print("1. Update your router/DNS to point this hostname to this machine:")
    print(f"   {TARGET_HOST} → <this machine's IP>")
    print("\n2. Power cycle or reconnect your GCP2.NET device")
    print("\n3. Wait for connection attempt...\n")
    print("If device connects → cert validation DISABLED → local MITM works!")
    print("If device fails → cert validation ENABLED → need AWS proxy")
    print("="*60 + "\n")

    try:
        while True:
            client_socket, addr = server.accept()

            try:
                # Try to establish TLS
                client_ssl = context.wrap_socket(client_socket, server_side=True)
                thread = threading.Thread(target=handle_connection, args=(client_ssl, addr))
                thread.start()
            except ssl.SSLError as e:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n[{timestamp}] ✗ TLS handshake failed from {addr}")
                print(f"[{timestamp}] → Error: {e}")
                print(f"[{timestamp}] → Device appears to VALIDATE certificates")
                print("\nConclusion: Need AWS-based proxy for decryption")
                client_socket.close()

    except KeyboardInterrupt:
        print("\n\nTest stopped.")
    finally:
        server.close()

if __name__ == "__main__":
    main()
