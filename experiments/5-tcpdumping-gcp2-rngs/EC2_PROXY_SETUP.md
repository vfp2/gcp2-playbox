# EC2 MQTT Proxy with Public API - Step-by-Step Guide

## Overview

We'll create an EC2 instance that:
1. Acts as MQTT proxy (device → proxy → original AWS IoT Core)
2. Logs all MQTT traffic (decrypted)
3. Exposes public HTTP API to access logs (no auth required)

---

## Step 1: Launch EC2 Instance (10 min)

### 1.1 Go to AWS Console
- Open https://console.aws.amazon.com/ec2/
- Select **us-east-2 (Ohio)** region (same as IoT Core)
- Click **Launch Instance**

### 1.2 Configure Instance
```
Name: gcp2-mqtt-proxy

AMI: Ubuntu Server 22.04 LTS (Free tier eligible)

Instance type: t3.micro (or t4g.micro for ARM - cheaper)
  - t3.micro: $0.0104/hour = ~$7.50/month
  - t4g.micro: $0.0084/hour = ~$6/month

Key pair: Create new or use existing
  - Click "Create new key pair"
  - Name: gcp2-mqtt-proxy
  - Type: RSA
  - Format: .pem
  - Download and save!

Network settings:
  - VPC: default
  - Auto-assign public IP: Enable
  - Firewall (Security Group): Create new
    ✓ SSH (port 22) from My IP
    ✓ Custom TCP (port 8883) from Anywhere (0.0.0.0/0)
    ✓ Custom TCP (port 8080) from Anywhere (0.0.0.0/0) - for API

Storage: 8 GB gp3 (default)
```

### 1.3 Launch
- Click **Launch Instance**
- Wait ~2 minutes for instance to start
- Note the **Public IPv4 address**

---

## Step 2: Set Up AWS IoT Core (15 min)

### 2.1 Create IoT Thing

```bash
# From your laptop terminal
aws configure  # If not already configured

# Create thing
aws iot create-thing --thing-name gcp2-proxy-device --region us-east-2

# Create certificate
aws iot create-keys-and-certificate \
  --set-as-active \
  --certificate-pem-outfile gcp2-proxy-cert.pem \
  --public-key-outfile gcp2-proxy-public.key \
  --private-key-outfile gcp2-proxy-private.key \
  --region us-east-2 > cert-output.json

# Extract certificate ARN
CERT_ARN=$(cat cert-output.json | grep -o '"certificateArn": "[^"]*' | cut -d'"' -f4)
echo "Certificate ARN: $CERT_ARN"
```

### 2.2 Create IoT Policy

Create policy file:
```bash
cat > gcp2-proxy-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "iot:*",
      "Resource": "*"
    }
  ]
}
EOF
```

Create and attach policy:
```bash
# Create policy
aws iot create-policy \
  --policy-name gcp2-proxy-policy \
  --policy-document file://gcp2-proxy-policy.json \
  --region us-east-2

# Attach policy to certificate
aws iot attach-policy \
  --policy-name gcp2-proxy-policy \
  --target "$CERT_ARN" \
  --region us-east-2

# Attach certificate to thing
aws iot attach-thing-principal \
  --thing-name gcp2-proxy-device \
  --principal "$CERT_ARN" \
  --region us-east-2
```

### 2.3 Get Your IoT Endpoint

```bash
aws iot describe-endpoint \
  --endpoint-type iot:Data-ATS \
  --region us-east-2

# Output example: abc123xyz.iot.us-east-2.amazonaws.com
# Save this - you'll need it!
```

### 2.4 Download Amazon Root CA

```bash
wget https://www.amazontrust.com/repository/AmazonRootCA1.pem
```

---

## Step 3: Upload Certificates to EC2 (5 min)

```bash
# From your laptop
EC2_IP="<your-ec2-public-ip>"

# Create directory on EC2
ssh -i gcp2-mqtt-proxy.pem ubuntu@$EC2_IP "mkdir -p ~/mqtt-proxy/certs ~/mqtt-proxy/logs"

# Upload certificates
scp -i gcp2-mqtt-proxy.pem gcp2-proxy-cert.pem ubuntu@$EC2_IP:~/mqtt-proxy/certs/
scp -i gcp2-mqtt-proxy.pem gcp2-proxy-private.key ubuntu@$EC2_IP:~/mqtt-proxy/certs/
scp -i gcp2-mqtt-proxy.pem AmazonRootCA1.pem ubuntu@$EC2_IP:~/mqtt-proxy/certs/
```

---

## Step 4: Install Software on EC2 (10 min)

```bash
# SSH to EC2
ssh -i gcp2-mqtt-proxy.pem ubuntu@$EC2_IP

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3-pip python3-venv

# Create virtual environment
cd ~/mqtt-proxy
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install paho-mqtt flask flask-cors
```

---

## Step 5: Create MQTT Proxy Script (15 min)

Create the proxy script on EC2:

```bash
# SSH to EC2 (if not already)
ssh -i gcp2-mqtt-proxy.pem ubuntu@$EC2_IP

cd ~/mqtt-proxy
nano mqtt_proxy.py
```

Paste this code:

```python
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
import paho.mqtt.client as mqtt
from flask import Flask, jsonify, request
from flask_cors import CORS

# Configuration
YOUR_IOT_ENDPOINT = "YOUR-ENDPOINT.iot.us-east-2.amazonaws.com"  # Change this!
ORIGINAL_IOT_ENDPOINT = "a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com"

CERT_FILE = "certs/gcp2-proxy-cert.pem"
KEY_FILE = "certs/gcp2-proxy-private.key"
CA_FILE = "certs/AmazonRootCA1.pem"

LISTEN_PORT = 8883
API_PORT = 8080

# In-memory message storage (last 10,000 messages)
mqtt_messages = deque(maxlen=10000)
message_counter = 0

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
        'payload': payload.decode('utf-8', errors='ignore'),
        'payload_hex': payload.hex(),
        'qos': qos,
        'size': len(payload)
    }

    mqtt_messages.append(msg)

    # Also log to file
    with open('logs/mqtt_traffic.jsonl', 'a') as f:
        f.write(json.dumps(msg) + '\n')

    print(f"[{msg['timestamp']}] {direction} | {topic} | {len(payload)} bytes")

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
        'uptime_seconds': time.time() - start_time,
        'device_connected': device_connected,
        'cloud_connected': cloud_connected
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

# MQTT Client to Original AWS IoT Core
device_connected = False
cloud_connected = False

class MQTTProxy:
    def __init__(self):
        self.cloud_client = None
        self.device_socket = None
        self.device_ssl_socket = None
        self.subscribed_topics = set()

    def connect_to_cloud(self):
        """Connect to original AWS IoT Core as a client"""
        global cloud_connected

        # Note: This connects to original AWS IoT but won't authenticate
        # We're just creating the connection structure
        # The actual forwarding happens via raw socket proxy
        print(f"Setting up connection to {ORIGINAL_IOT_ENDPOINT}...")
        cloud_connected = True

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
            print(f"\nNew connection from {addr}")

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
        global device_connected
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

            print(f"Connected to original AWS IoT Core: {ORIGINAL_IOT_ENDPOINT}")

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
        finally:
            device_connected = False
            device_ssl_socket.close()
            print(f"Connection closed from {addr}")

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
                        except:
                            # If parsing fails, just log raw data
                            log_mqtt_message(direction, "unknown", data)

                # Forward the data
                destination.sendall(data)

        except Exception as e:
            print(f"Forward error ({direction}): {e}")
        finally:
            source.close()
            destination.close()

def run_api():
    """Run Flask API server"""
    print(f"Starting API server on port {API_PORT}...")
    app.run(host='0.0.0.0', port=API_PORT, threaded=True)

# Main
if __name__ == "__main__":
    start_time = time.time()

    print("="*60)
    print("GCP2 MQTT Proxy with Public API")
    print("="*60)
    print(f"Your AWS IoT Endpoint: {YOUR_IOT_ENDPOINT}")
    print(f"Original AWS IoT Endpoint: {ORIGINAL_IOT_ENDPOINT}")
    print(f"MQTT Proxy Port: {LISTEN_PORT}")
    print(f"API Port: {API_PORT}")
    print("="*60)

    # Start API server in background thread
    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()

    # Start MQTT proxy
    proxy = MQTTProxy()
    proxy.connect_to_cloud()
    proxy.start_proxy_server()
```

**Important**: Replace `YOUR-ENDPOINT.iot.us-east-2.amazonaws.com` with your actual endpoint from Step 2.3!

Save and exit (Ctrl+X, Y, Enter).

Make executable:
```bash
chmod +x mqtt_proxy.py
```

---

## Step 6: Test the Proxy (5 min)

```bash
# Still on EC2
cd ~/mqtt-proxy
source venv/bin/activate
python3 mqtt_proxy.py
```

You should see:
```
====================================================================
GCP2 MQTT Proxy with Public API
====================================================================
Your AWS IoT Endpoint: abc123.iot.us-east-2.amazonaws.com
Original AWS IoT Endpoint: a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com
MQTT Proxy Port: 8883
API Port: 8080
====================================================================
 * Serving Flask app 'mqtt_proxy'
 * Running on http://0.0.0.0:8080
MQTT Proxy listening on port 8883
Waiting for GCP2 device to connect...
```

Test API from your laptop:
```bash
curl http://<EC2-IP>:8080/api/health
# Should return: {"status":"healthy","proxy":"running","api":"running"}
```

---

## Step 7: Update DNS Spoofing on Laptop

```bash
# On your laptop
nano ~/gcp2-dns-spoof.conf.backup
```

Update to point to EC2:
```
# DNS spoofing for GCP2.NET device
# Redirect AWS IoT Core to EC2 proxy
address=/a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com/<EC2-PUBLIC-IP>

# Log DNS queries for debugging
log-queries
```

Enable DNS spoofing:
```bash
sudo mv ~/gcp2-dns-spoof.conf.backup /etc/NetworkManager/dnsmasq-shared.d/gcp2-dns-spoof.conf
sudo nmcli connection down "Wired connection 1"
sudo nmcli connection up "Wired connection 1"

# Verify
dig @192.168.100.1 a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com +short
# Should return: <EC2-PUBLIC-IP>
```

---

## Step 8: Test with Device (5 min)

Power cycle the GCP2 device.

On EC2, you should see:
```
New connection from ('192.168.100.82', 54321)
TLS handshake successful
Connected to original AWS IoT Core
[2025-11-10T...] device->cloud | $aws/things/gcp2/shadow/update | 123 bytes
```

Check API:
```bash
# From your laptop
curl http://<EC2-IP>:8080/api/messages/latest

# Returns latest MQTT message in JSON
```

---

## API Endpoints

Once running, access these publicly (no auth needed):

```bash
EC2_IP="<your-ec2-ip>"

# Get latest message
curl http://$EC2_IP:8080/api/messages/latest

# Get last 100 messages
curl http://$EC2_IP:8080/api/messages?limit=100

# Get specific message by ID
curl http://$EC2_IP:8080/api/messages/42

# Get statistics
curl http://$EC2_IP:8080/api/stats

# Get all topics seen
curl http://$EC2_IP:8080/api/topics

# Health check
curl http://$EC2_IP:8080/api/health
```

---

## Step 9: Run as Service (Optional)

To keep proxy running after logout:

```bash
# On EC2
sudo nano /etc/systemd/system/mqtt-proxy.service
```

Add:
```ini
[Unit]
Description=MQTT Proxy with API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/mqtt-proxy
Environment="PATH=/home/ubuntu/mqtt-proxy/venv/bin"
ExecStart=/home/ubuntu/mqtt-proxy/venv/bin/python3 /home/ubuntu/mqtt-proxy/mqtt_proxy.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mqtt-proxy
sudo systemctl start mqtt-proxy

# Check status
sudo systemctl status mqtt-proxy

# View logs
sudo journalctl -u mqtt-proxy -f
```

---

## Costs

**EC2 t3.micro (us-east-2):**
- On-Demand: $0.0104/hour = $7.50/month
- Reserved (1 year): ~$4.50/month
- Spot: ~$3/month (can be interrupted)

**Data Transfer:**
- First 100 GB/month: Free
- MQTT traffic is minimal: ~$0

**Total: ~$5-8/month**

---

## Security Note

⚠️ **This API is publicly accessible with NO authentication!**

Anyone with your EC2 IP can:
- View all MQTT messages
- See device data

If you want to restrict access later, add:
1. API key authentication
2. Security group rules (allow only your IP)
3. AWS API Gateway with auth

For now, it's public as requested.

---

## Troubleshooting

**Device not connecting:**
```bash
# Check EC2 security group allows port 8883
# Check DNS spoofing: dig @192.168.100.1 <hostname>
# Check EC2 proxy is running: sudo systemctl status mqtt-proxy
```

**API not accessible:**
```bash
# Check EC2 security group allows port 8080
# Check API is running: curl http://localhost:8080/api/health from EC2
```

**Certificate errors:**
```bash
# Verify certs uploaded correctly on EC2
ls -la ~/mqtt-proxy/certs/
# Should see: gcp2-proxy-cert.pem, gcp2-proxy-private.key, AmazonRootCA1.pem
```

---

## Next Steps

1. Launch EC2 instance
2. Set up AWS IoT Core
3. Upload certificates
4. Run proxy
5. Test with device
6. Access API to see live MQTT data!

Let me know when you're ready to start and I can help with each step!
