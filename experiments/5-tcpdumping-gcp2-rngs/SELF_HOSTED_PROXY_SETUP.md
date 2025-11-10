# Self-Hosted MQTT Proxy Setup with AWS IoT Core

## Architecture

```
┌─────────────┐  DNS Spoof   ┌──────────────────┐
│ GCP2 Device │──────────────▶│  Your AWS IoT    │
│             │  TLS (8883)   │  Core Endpoint   │
└─────────────┘               └──────────────────┘
                                      │
                                      │ IoT Rule (HTTP action)
                                      │ sends all MQTT messages
                                      ▼
                              ┌──────────────────┐
                              │  Self-Hosted     │◀──┐
                              │  MQTT Bridge     │   │
                              │  (Your Server)   │   │
                              └──────────────────┘   │
                                      │              │
                                      │ Logs MQTT    │ Subscribes
                                      │ traffic      │ to topics
                                      │              │
                                      ▼              │
                              ┌──────────────────────┴────┐
                              │  Original AWS IoT Core    │
                              │  a1vfh7jker84ic-ats...    │
                              └───────────────────────────┘
```

## How It Works

1. **Device → Your AWS IoT Core**: Device connects using valid AWS certificate
2. **AWS IoT Rule**: Forwards ALL incoming messages to your server via HTTP webhook
3. **Your Server**:
   - Receives HTTP webhooks with decrypted MQTT data
   - Logs everything
   - Acts as MQTT client to original AWS IoT Core
   - Subscribes to response topics and forwards back to your AWS IoT Core
4. **Bidirectional Communication**: Your server bridges both directions

## Problems with This Approach

❌ **AWS IoT Rules are one-way**: Device→Server messages work, but Server→Device requires complex setup

❌ **Subscription handling**: Device might subscribe to topics that don't exist on your AWS IoT Core

❌ **Connection state**: Original server might reject republished messages (wrong client ID, certs, etc.)

---

## Better Solution: Run AWS IoT Device SDK Proxy on Your Server

Instead of using IoT Rules, run the proxy entirely on AWS (small EC2) or your server with AWS credentials.

### Option A: EC2-Based MQTT Proxy (Recommended)

**Why EC2?** AWS gives you valid certificates AND low latency to original AWS IoT Core.

```
GCP2 Device → DNS Spoof → EC2 MQTT Proxy → Original AWS IoT Core
             (laptop)      (valid AWS cert)  (real destination)
                               │
                               └──▶ Streams logs to your server
                                    via SSH tunnel or S3
```

**Setup:**

1. **Launch tiny EC2 instance** (t4g.nano - $0.0042/hr = $3/month)
2. **Install MQTT proxy** that:
   - Listens on 8883 with AWS IoT certificate
   - Forwards all traffic to original AWS IoT Core
   - Logs to file
3. **Stream logs** to your server:
   ```bash
   # SSH tunnel from your server
   ssh -R 9999:localhost:9999 ubuntu@ec2-ip

   # Or sync logs periodically
   rsync -avz ubuntu@ec2-ip:~/mqtt-logs/ ./logs/
   ```

---

## Option B: Self-Hosted with AWS IoT Core Certificate (Your Preference)

If you want everything on your self-hosted server, you can still use AWS IoT Core certificates:

### Architecture
```
GCP2 Device → DNS Spoof → Your Server (8883) → Original AWS IoT Core
             (laptop)      (AWS IoT cert)       (real destination)
```

### Challenge
The device validates the certificate hostname. AWS won't give you a cert for `a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com` - you can only get certs for YOUR AWS IoT endpoint.

### Solution: Two DNS Spoofs

**Step 1**: Point original hostname to YOUR AWS IoT endpoint hostname:
```bash
# DNS spoof on laptop
address=/a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com/abc123.iot.us-east-2.amazonaws.com
```

**Step 2**: Point YOUR AWS IoT endpoint hostname to your self-hosted server:
```bash
# Public DNS (Route53 or your registrar) - WON'T WORK
# AWS controls *.iot.us-east-2.amazonaws.com, you can't override this
```

❌ **Problem**: You can't create DNS records for `*.amazonaws.com` domains

---

## Option C: Use AWS IoT Core as Pure Passthrough (Best for Your Use Case)

Use AWS IoT Core ONLY for the TLS certificate, then immediately forward to your server:

### Step-by-Step

#### 1. Set Up AWS IoT Core (15 min)

```bash
# Install AWS CLI
pip3 install awscli

# Configure AWS credentials
aws configure
```

**Create IoT Thing:**
```bash
aws iot create-thing --thing-name gcp2-proxy-device

# Create certificate
aws iot create-keys-and-certificate \
  --set-as-active \
  --certificate-pem-outfile cert.pem \
  --public-key-outfile public.key \
  --private-key-outfile private.key

# Note the certificate ARN from output
```

**Create IoT Policy** (allows all for testing):
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "iot:*",
    "Resource": "*"
  }]
}
```

```bash
aws iot create-policy \
  --policy-name gcp2-proxy-policy \
  --policy-document file://policy.json

# Attach policy to certificate
aws iot attach-policy \
  --policy-name gcp2-proxy-policy \
  --target <CERTIFICATE-ARN>

# Attach certificate to thing
aws iot attach-thing-principal \
  --thing-name gcp2-proxy-device \
  --principal <CERTIFICATE-ARN>

# Get your IoT endpoint
aws iot describe-endpoint --endpoint-type iot:Data-ATS
# Output: abc123.iot.us-east-2.amazonaws.com
```

#### 2. Create IoT Rule to Forward to Your Server

**Create HTTP Rule:**
```bash
# Create rule that forwards ALL messages to your server
aws iot create-topic-rule \
  --rule-name ForwardToSelfHosted \
  --topic-rule-payload file://rule.json
```

**rule.json:**
```json
{
  "sql": "SELECT * FROM '#'",
  "description": "Forward all MQTT messages to self-hosted server",
  "actions": [{
    "http": {
      "url": "https://your-server.com:8443/mqtt-webhook",
      "confirmationUrl": "https://your-server.com:8443/mqtt-webhook",
      "headers": [{
        "key": "X-Auth-Token",
        "value": "your-secret-token"
      }]
    }
  }],
  "ruleDisabled": false,
  "awsIotSqlVersion": "2016-03-23"
}
```

#### 3. Set Up Your Self-Hosted Server

**Install dependencies:**
```bash
# On your server
pip3 install flask paho-mqtt
```

**Create MQTT bridge** (`mqtt_bridge.py`):
```python
#!/usr/bin/env python3
"""
MQTT Bridge: Receives from AWS IoT Core, logs, forwards to original
"""
from flask import Flask, request, jsonify
import paho.mqtt.client as mqtt
import ssl
import json
import logging
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('mqtt_traffic.log'),
        logging.StreamHandler()
    ]
)

# Original AWS IoT Core connection
ORIGINAL_IOT_ENDPOINT = "a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com"
ORIGINAL_IOT_PORT = 8883

# Download Amazon Root CA
# wget https://www.amazontrust.com/repository/AmazonRootCA1.pem

# MQTT client to forward to original AWS IoT Core
# NOTE: This won't work without valid credentials for the original endpoint
# You'd need to use basic MQTT without AWS IoT Core authentication
forward_client = None

def setup_forward_client():
    """Set up MQTT client to forward to original AWS IoT Core"""
    global forward_client

    # This is the challenge: we don't have credentials for the original endpoint
    # You may need to use standard MQTT broker instead
    # For now, just log
    logging.info("Forward client setup - would connect to original AWS IoT")

@app.route('/mqtt-webhook', methods=['POST'])
def mqtt_webhook():
    """Receive MQTT messages from AWS IoT Core Rule"""
    try:
        data = request.get_json()

        # Log the message
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'topic': data.get('topic', 'unknown'),
            'message': data,
            'headers': dict(request.headers)
        }

        logging.info(f"MQTT Message: {json.dumps(log_entry, indent=2)}")

        # Write to dedicated log file
        with open('mqtt_messages.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Forward to original AWS IoT Core
        # NOTE: This requires valid credentials for the original endpoint
        # which we don't have access to

        return jsonify({'status': 'received'}), 200

    except Exception as e:
        logging.error(f"Error processing webhook: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    setup_forward_client()

    # Run Flask server with TLS
    # You'll need SSL cert for your domain
    app.run(
        host='0.0.0.0',
        port=8443,
        ssl_context=('cert.pem', 'key.pem')  # Your server's SSL cert
    )
```

**Run the server:**
```bash
python3 mqtt_bridge.py
```

#### 4. Enable DNS Spoofing on Laptop

```bash
# Edit DNS config
sudo nano /etc/NetworkManager/dnsmasq-shared.d/gcp2-dns-spoof.conf

# Point to YOUR AWS IoT endpoint
address=/a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com/<YOUR-IOT-ENDPOINT-IP>

# Restart network
sudo nmcli connection down "Wired connection 1"
sudo nmcli connection up "Wired connection 1"
```

---

## The Core Problem

**You can't forward to the original AWS IoT Core** because:
1. You don't have the device's AWS IoT credentials
2. The original AWS IoT Core expects specific client certs/authentication
3. Simply republishing messages will fail authentication

## Alternative: Log Only (No Forward)

If you just want to **log** the traffic without forwarding:

1. Use your AWS IoT Core as the endpoint (device connects there)
2. Use IoT Rules to forward to your server for logging
3. **Don't try to forward to original AWS IoT Core**
4. Accept that the device's backend won't receive the data

This is simpler and works if you just want to analyze the MQTT protocol.

---

## Recommended Approach

Given the constraints, I recommend:

### Hybrid: EC2 Proxy + Log Streaming

1. **Run MQTT proxy on EC2** (t4g.nano - $3/month)
2. **Proxy does bidirectional MQTT** between device and original AWS IoT Core
3. **Stream logs to your server** via:
   - SSH tunnel (real-time)
   - S3 bucket (batch)
   - Direct TCP stream

**Benefits:**
- EC2 has low latency to AWS IoT Core
- Valid AWS certificate works
- Your server gets all the logs
- Original service keeps working

**Would you like me to create the EC2-based solution, or do you have another approach in mind?**
