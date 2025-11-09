#!/usr/bin/env python3
"""
AWS IoT Proxy - Runs on AWS EC2, streams logs to local machine
This terminates TLS with valid AWS cert, logs plaintext, forwards to real IoT Core
"""
import socket
import ssl
import threading
import json
import struct
from datetime import datetime

# Configuration
LISTEN_PORT = 8883
FORWARD_HOST = 'a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com'
FORWARD_PORT = 8883

# AWS IoT Core certificate paths (on EC2 instance)
# You'd get these from your own AWS IoT Core endpoint
CERT_FILE = '/path/to/your-iot-endpoint-cert.pem'
KEY_FILE = '/path/to/your-iot-endpoint-key.pem'

# For local logging - could be a socket connection to your local machine
LOG_FILE = 'mqtt_traffic.log'

def parse_mqtt_packet(data):
    """Parse MQTT packet to extract basic info"""
    if len(data) < 2:
        return None

    try:
        # MQTT fixed header
        msg_type = (data[0] >> 4) & 0x0F
        types = {
            1: 'CONNECT', 2: 'CONNACK', 3: 'PUBLISH', 4: 'PUBACK',
            5: 'PUBREC', 6: 'PUBREL', 7: 'PUBCOMP', 8: 'SUBSCRIBE',
            9: 'SUBACK', 10: 'UNSUBSCRIBE', 11: 'UNSUBACK',
            12: 'PINGREQ', 13: 'PINGRESP', 14: 'DISCONNECT'
        }

        packet_type = types.get(msg_type, f'UNKNOWN({msg_type})')

        # Get remaining length
        multiplier = 1
        value = 0
        pos = 1
        while True:
            if pos >= len(data):
                break
            byte = data[pos]
            value += (byte & 127) * multiplier
            multiplier *= 128
            pos += 1
            if (byte & 128) == 0:
                break

        # Try to extract topic for PUBLISH packets
        topic = None
        if msg_type == 3 and pos + 2 < len(data):  # PUBLISH
            topic_len = (data[pos] << 8) | data[pos + 1]
            if pos + 2 + topic_len <= len(data):
                topic = data[pos + 2:pos + 2 + topic_len].decode('utf-8', errors='ignore')

        return {
            'type': packet_type,
            'length': value,
            'topic': topic
        }
    except Exception as e:
        return {'type': 'PARSE_ERROR', 'error': str(e)}

def log_traffic(direction, data, conn_id):
    """Log decrypted MQTT traffic"""
    timestamp = datetime.now().isoformat()
    parsed = parse_mqtt_packet(data)

    log_entry = {
        'timestamp': timestamp,
        'connection_id': conn_id,
        'direction': direction,
        'size': len(data),
        'mqtt': parsed,
        'hex': data[:64].hex() if len(data) <= 64 else data[:64].hex() + '...'
    }

    # Write to log file
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    # Print to console
    if parsed and parsed.get('topic'):
        print(f"[{timestamp}] [{conn_id}] {direction} {parsed['type']} topic={parsed['topic']} ({len(data)} bytes)")
    elif parsed:
        print(f"[{timestamp}] [{conn_id}] {direction} {parsed['type']} ({len(data)} bytes)")
    else:
        print(f"[{timestamp}] [{conn_id}] {direction} ({len(data)} bytes)")

def forward_data(source, destination, direction, conn_id):
    """Forward data and log it"""
    try:
        while True:
            data = source.recv(4096)
            if not data:
                break

            log_traffic(direction, data, conn_id)
            destination.sendall(data)
    except Exception as e:
        print(f"[{conn_id}] Error in {direction}: {e}")
    finally:
        source.close()
        destination.close()

def handle_client(client_socket, addr, conn_id):
    """Handle client connection with TLS termination"""
    print(f"\n[{conn_id}] Connection from {addr}")

    try:
        # Connect to real AWS IoT Core
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.connect((FORWARD_HOST, FORWARD_PORT))

        # Wrap server connection in TLS (as client to real AWS)
        context = ssl.create_default_context()
        server_ssl = context.wrap_socket(server_socket, server_hostname=FORWARD_HOST)

        print(f"[{conn_id}] Connected to {FORWARD_HOST}")

        # Forward bidirectionally
        t1 = threading.Thread(target=forward_data,
                            args=(client_socket, server_ssl, "CLIENT->SERVER", conn_id))
        t2 = threading.Thread(target=forward_data,
                            args=(server_ssl, client_socket, "SERVER->CLIENT", conn_id))

        t1.daemon = True
        t2.daemon = True
        t1.start()
        t2.start()

        t1.join()
        t2.join()

    except Exception as e:
        print(f"[{conn_id}] Error: {e}")
    finally:
        client_socket.close()

def main():
    # Create SSL context with YOUR AWS IoT endpoint certificate
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)

    # Create listening socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', LISTEN_PORT))
    server.listen(5)

    print(f"AWS IoT Proxy listening on port {LISTEN_PORT}")
    print(f"Forwarding to {FORWARD_HOST}:{FORWARD_PORT}")
    print(f"Logging to {LOG_FILE}\n")

    conn_counter = 0
    try:
        while True:
            client_socket, addr = server.accept()
            conn_counter += 1
            conn_id = f"CONN-{conn_counter:04d}"

            # Wrap in SSL with YOUR cert
            client_ssl = context.wrap_socket(client_socket, server_side=True)

            thread = threading.Thread(target=handle_client,
                                    args=(client_ssl, addr, conn_id))
            thread.daemon = True
            thread.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.close()

if __name__ == "__main__":
    main()
