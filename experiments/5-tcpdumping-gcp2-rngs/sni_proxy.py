#!/usr/bin/env python3
"""
SNI Proxy - Forwards TLS traffic without decryption
Captures encrypted traffic and metadata but preserves end-to-end encryption
"""
import socket
import threading
import time
from datetime import datetime

LISTEN_HOST = '0.0.0.0'
LISTEN_PORT = 8883
REMOTE_HOST = 'a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com'
REMOTE_PORT = 8883

def extract_sni(data):
    """Extract SNI from TLS Client Hello"""
    try:
        if len(data) < 43:
            return None
        # Check for TLS handshake
        if data[0] != 0x16:  # Handshake
            return None
        # Skip to handshake type
        if data[5] != 0x01:  # Client Hello
            return None

        # Parse extensions (simplified)
        pos = 43  # Skip fixed header
        session_id_len = data[pos]
        pos += 1 + session_id_len

        if pos + 2 > len(data):
            return None
        cipher_suites_len = (data[pos] << 8) | data[pos + 1]
        pos += 2 + cipher_suites_len

        if pos >= len(data):
            return None
        compression_len = data[pos]
        pos += 1 + compression_len

        # Extensions
        if pos + 2 > len(data):
            return None
        extensions_len = (data[pos] << 8) | data[pos + 1]
        pos += 2

        end = pos + extensions_len
        while pos + 4 < end:
            ext_type = (data[pos] << 8) | data[pos + 1]
            ext_len = (data[pos + 2] << 8) | data[pos + 3]
            pos += 4

            if ext_type == 0:  # SNI
                list_len = (data[pos] << 8) | data[pos + 1]
                pos += 2
                name_type = data[pos]
                pos += 1
                name_len = (data[pos] << 8) | data[pos + 1]
                pos += 2
                sni = data[pos:pos + name_len].decode('utf-8', errors='ignore')
                return sni
            pos += ext_len
    except:
        pass
    return None

def forward(source, destination, direction, conn_id):
    """Forward data between source and destination"""
    try:
        first_packet = True
        while True:
            data = source.recv(4096)
            if not data:
                break

            # Log packet info
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"[{timestamp}] [{conn_id}] {direction}: {len(data)} bytes")

            # Extract SNI from first client packet
            if first_packet and direction == "CLIENT->SERVER":
                sni = extract_sni(data)
                if sni:
                    print(f"[{timestamp}] [{conn_id}] SNI: {sni}")
                first_packet = False

            destination.sendall(data)
    except Exception as e:
        print(f"[{conn_id}] Forward error ({direction}): {e}")
    finally:
        source.close()
        destination.close()

def handle_connection(client_socket, addr, conn_id):
    """Handle a single client connection"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"\n[{timestamp}] [{conn_id}] New connection from {addr[0]}:{addr[1]}")

    try:
        # Connect to real server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.connect((REMOTE_HOST, REMOTE_PORT))
        print(f"[{timestamp}] [{conn_id}] Connected to {REMOTE_HOST}:{REMOTE_PORT}")

        # Create bidirectional forwarding
        client_to_server = threading.Thread(
            target=forward,
            args=(client_socket, server_socket, "CLIENT->SERVER", conn_id)
        )
        server_to_client = threading.Thread(
            target=forward,
            args=(server_socket, client_socket, "SERVER->CLIENT", conn_id)
        )

        client_to_server.daemon = True
        server_to_client.daemon = True

        client_to_server.start()
        server_to_client.start()

        client_to_server.join()
        server_to_client.join()

    except Exception as e:
        print(f"[{conn_id}] Connection error: {e}")
    finally:
        client_socket.close()
        print(f"[{conn_id}] Connection closed")

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_HOST, LISTEN_PORT))
    server.listen(5)

    print(f"SNI Proxy listening on {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"Forwarding to {REMOTE_HOST}:{REMOTE_PORT}")
    print(f"Ready to capture traffic...\n")

    conn_counter = 0
    try:
        while True:
            client_socket, addr = server.accept()
            conn_counter += 1
            conn_id = f"CONN-{conn_counter:04d}"

            thread = threading.Thread(
                target=handle_connection,
                args=(client_socket, addr, conn_id)
            )
            thread.daemon = True
            thread.start()
    except KeyboardInterrupt:
        print("\nShutting down proxy...")
    finally:
        server.close()

if __name__ == "__main__":
    main()
