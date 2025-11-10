#!/usr/bin/env python3
"""
MQTT Proxy with Public API
- Accepts connections from GCP2 device with AWS IoT certificate
- Forwards to original AWS IoT Core
- Logs all MQTT traffic
- Exposes HTTP API to access logs
"""

import ssl
import socket
import threading
import json
import time
from datetime import datetime
from collections import deque
from flask import Flask, jsonify, request
from flask_cors import CORS

# Configuration
YOUR_IOT_ENDPOINT = "REPLACE_WITH_YOUR_ENDPOINT.iot.us-east-2.amazonaws.com"  # ← CHANGE THIS!
ORIGINAL_IOT_ENDPOINT = "a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com"

CERT_FILE = "certs/gcp2-proxy-cert.pem"
KEY_FILE = "certs/gcp2-proxy-private.key"
CA_FILE = "certs/AmazonRootCA1.pem"

LISTEN_PORT = 8883
API_PORT = 8080

# In-memory message storage (last 10,000 messages)
mqtt_messages = deque(maxlen=10000)
message_counter = 0
start_time = time.time()

# Connection status
device_connected = False
cloud_connected = False

# Flask API
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def log_mqtt_message(direction, topic, payload, qos=0):
    """Log MQTT message and store in memory"""
    global message_counter
    message_counter += 1

    msg = {
        'id': message_counter,
        'timestamp': datetime.now().isoformat(),
        'direction': direction,  # 'device->cloud' or 'cloud->device'
        'topic': topic,
        'payload': payload.decode('utf-8', errors='ignore') if payload else '',
        'payload_hex': payload.hex() if payload else '',
        'qos': qos,
        'size': len(payload) if payload else 0
    }

    mqtt_messages.append(msg)

    # Also log to file
    with open('logs/mqtt_traffic.jsonl', 'a') as f:
        f.write(json.dumps(msg) + '\n')

    print(f"[{msg['timestamp']}] {direction} | {topic} | {len(payload) if payload else 0} bytes")

# Flask API Routes
@app.route('/api/messages', methods=['GET'])
def get_messages():
    """Get all logged MQTT messages"""
    limit = request.args.get('limit', type=int, default=100)
    offset = request.args.get('offset', type=int, default=0)

    messages = list(mqtt_messages)
    messages.reverse()  # Newest first

    return jsonify({
        'total': len(mqtt_messages),
        'limit': limit,
        'offset': offset,
        'messages': messages[offset:offset+limit]
    })

@app.route('/api/messages/latest', methods=['GET'])
def get_latest():
    """Get latest MQTT message"""
    if mqtt_messages:
        return jsonify(mqtt_messages[-1])
    return jsonify({'error': 'No messages yet'}), 404

@app.route('/api/messages/<int:msg_id>', methods=['GET'])
def get_message(msg_id):
    """Get specific message by ID"""
    for msg in mqtt_messages:
        if msg['id'] == msg_id:
            return jsonify(msg)
    return jsonify({'error': 'Message not found'}), 404

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get proxy statistics"""
    return jsonify({
        'total_messages': message_counter,
        'messages_in_memory': len(mqtt_messages),
        'uptime_seconds': int(time.time() - start_time),
        'device_connected': device_connected,
        'cloud_connected': cloud_connected,
        'your_iot_endpoint': YOUR_IOT_ENDPOINT,
        'original_iot_endpoint': ORIGINAL_IOT_ENDPOINT
    })

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get list of all topics seen"""
    topics = {}
    for msg in mqtt_messages:
        topic = msg['topic']
        if topic not in topics:
            topics[topic] = 0
        topics[topic] += 1

    return jsonify({
        'topics': [
            {'topic': t, 'count': c}
            for t, c in sorted(topics.items(), key=lambda x: x[1], reverse=True)
        ]
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'proxy': 'running',
        'api': 'running'
    })

@app.route('/', methods=['GET'])
def index():
    """Simple web UI"""
    return """
    <html>
    <head><title>GCP2 MQTT Proxy</title></head>
    <body>
        <h1>GCP2 MQTT Proxy</h1>
        <p>Proxy is running!</p>
        <h2>API Endpoints:</h2>
        <ul>
            <li><a href="/api/health">/api/health</a> - Health check</li>
            <li><a href="/api/stats">/api/stats</a> - Statistics</li>
            <li><a href="/api/messages/latest">/api/messages/latest</a> - Latest message</li>
            <li><a href="/api/messages?limit=10">/api/messages?limit=10</a> - Last 10 messages</li>
            <li><a href="/api/topics">/api/topics</a> - All topics</li>
        </ul>
    </body>
    </html>
    """

class MQTTProxy:
    def __init__(self):
        self.subscribed_topics = set()

    def start_proxy_server(self):
        """Start TLS server that devices connect to"""
        # Create SSL context with YOUR AWS IoT certificate
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(CERT_FILE, KEY_FILE)

        # Create listening socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', LISTEN_PORT))
        server.listen(5)

        print(f"MQTT Proxy listening on port {LISTEN_PORT}")
        print(f"Waiting for GCP2 device to connect...")

        while True:
            client_socket, addr = server.accept()
            print(f"\n{'='*60}")
            print(f"New connection from {addr}")

            try:
                # Wrap in TLS
                client_ssl = context.wrap_socket(client_socket, server_side=True)
                print(f"TLS handshake successful with {addr}")

                # Handle this connection in a thread
                thread = threading.Thread(
                    target=self.handle_device_connection,
                    args=(client_ssl, addr)
                )
                thread.daemon = True
                thread.start()

            except Exception as e:
                print(f"Error accepting connection: {e}")
                client_socket.close()

    def handle_device_connection(self, device_ssl_socket, addr):
        """Handle bidirectional MQTT proxy"""
        global device_connected, cloud_connected
        device_connected = True

        print(f"Handling device connection from {addr}")

        try:
            # Connect to original AWS IoT Core
            cloud_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            cloud_socket.connect((ORIGINAL_IOT_ENDPOINT, 8883))

            # Wrap in TLS
            cloud_context = ssl.create_default_context()
            cloud_ssl = cloud_context.wrap_socket(
                cloud_socket,
                server_hostname=ORIGINAL_IOT_ENDPOINT
            )

            print(f"✓ Connected to original AWS IoT Core: {ORIGINAL_IOT_ENDPOINT}")
            cloud_connected = True

            # Start bidirectional forwarding
            device_to_cloud = threading.Thread(
                target=self.forward_data,
                args=(device_ssl_socket, cloud_ssl, "device->cloud", addr)
            )

            cloud_to_device = threading.Thread(
                target=self.forward_data,
                args=(cloud_ssl, device_ssl_socket, "cloud->device", addr)
            )

            device_to_cloud.daemon = True
            cloud_to_device.daemon = True

            device_to_cloud.start()
            cloud_to_device.start()

            device_to_cloud.join()
            cloud_to_device.join()

        except Exception as e:
            print(f"Error in device connection handler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            device_connected = False
            cloud_connected = False
            device_ssl_socket.close()
            print(f"{'='*60}")
            print(f"Connection closed from {addr}")
            print(f"{'='*60}\n")

    def forward_data(self, source, destination, direction, addr):
        """Forward data and log MQTT packets"""
        try:
            while True:
                data = source.recv(4096)
                if not data:
                    break

                # Parse MQTT packet (simplified)
                if len(data) >= 2:
                    msg_type = (data[0] >> 4) & 0x0F
                    msg_types = {
                        1: 'CONNECT', 2: 'CONNACK', 3: 'PUBLISH', 4: 'PUBACK',
                        5: 'PUBREC', 6: 'PUBREL', 7: 'PUBCOMP', 8: 'SUBSCRIBE',
                        9: 'SUBACK', 10: 'UNSUBSCRIBE', 11: 'UNSUBACK',
                        12: 'PINGREQ', 13: 'PINGRESP', 14: 'DISCONNECT'
                    }

                    # Try to extract topic and payload for PUBLISH messages
                    if msg_type == 3:  # PUBLISH
                        try:
                            # Parse MQTT PUBLISH packet
                            pos = 1
                            remaining_len = 0
                            multiplier = 1

                            while pos < len(data):
                                byte = data[pos]
                                remaining_len += (byte & 127) * multiplier
                                multiplier *= 128
                                pos += 1
                                if (byte & 128) == 0:
                                    break

                            # Extract topic
                            if pos + 2 <= len(data):
                                topic_len = (data[pos] << 8) | data[pos + 1]
                                pos += 2

                                if pos + topic_len <= len(data):
                                    topic = data[pos:pos + topic_len].decode('utf-8', errors='ignore')
                                    pos += topic_len

                                    # Payload is the rest
                                    payload = data[pos:]

                                    log_mqtt_message(direction, topic, payload)
                        except Exception as e:
                            # If parsing fails, just log raw data
                            log_mqtt_message(direction, f"parse_error_{msg_types.get(msg_type, 'UNKNOWN')}", data)
                    else:
                        # Log other MQTT packet types
                        packet_type = msg_types.get(msg_type, f'UNKNOWN({msg_type})')
                        log_mqtt_message(direction, f"control_{packet_type}", data)

                # Forward the data
                destination.sendall(data)

        except Exception as e:
            print(f"Forward error ({direction}): {e}")
        finally:
            try:
                source.close()
            except:
                pass
            try:
                destination.close()
            except:
                pass

def run_api():
    """Run Flask API server"""
    print(f"Starting API server on port {API_PORT}...")
    app.run(host='0.0.0.0', port=API_PORT, threaded=True)

# Main
if __name__ == "__main__":
    print("="*60)
    print("GCP2 MQTT Proxy with Public API")
    print("="*60)
    print(f"Your AWS IoT Endpoint: {YOUR_IOT_ENDPOINT}")
    print(f"Original AWS IoT Endpoint: {ORIGINAL_IOT_ENDPOINT}")
    print(f"MQTT Proxy Port: {LISTEN_PORT}")
    print(f"API Port: {API_PORT}")
    print("="*60)
    print()

    if "REPLACE_WITH_YOUR_ENDPOINT" in YOUR_IOT_ENDPOINT:
        print("⚠️  WARNING: Please edit mqtt_proxy.py and set YOUR_IOT_ENDPOINT!")
        print("⚠️  Get it with: aws iot describe-endpoint --endpoint-type iot:Data-ATS")
        print()

    # Start API server in background thread
    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()

    # Give API a moment to start
    time.sleep(1)

    # Start MQTT proxy
    proxy = MQTTProxy()
    proxy.start_proxy_server()
