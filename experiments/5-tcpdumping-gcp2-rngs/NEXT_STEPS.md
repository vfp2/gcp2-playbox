# Next Steps: AWS EC2 Proxy Setup

## Current Status (2025-11-10)

✅ **Completed:**
- Analyzed pcap and identified AWS IoT Core destination
- Set up local test network (USB Ethernet with DHCP/NAT/DNS)
- Tested certificate validation → **Device DOES validate certificates**
- Device can now connect normally to real AWS IoT Core through NAT

## What We Need Tomorrow: AWS-Based Proxy

### Why We Need This
The GCP2.NET device validates SSL certificates and rejects self-signed certs. To decrypt the MQTT traffic, we need a man-in-the-middle proxy that presents a valid AWS certificate.

### Architecture
```
GCP2 Device → DNS Spoof → Your AWS IoT Endpoint → Proxy on EC2 → Real AWS IoT Core
             (laptop)      (valid AWS cert)         (logs MQTT)   (original destination)
```

### Step-by-Step Plan

#### 1. Set Up AWS IoT Core in Your Account (~15 min)

**In AWS Console (us-east-2 region):**
1. Go to AWS IoT Core → Settings
2. Note your endpoint: `xxxxxxx.iot.us-east-2.amazonaws.com`
3. Create a "Thing" (device):
   - IoT Core → Manage → All devices → Things → Create thing
   - Name: `gcp2-proxy-test`
   - Auto-generate certificate
   - Download: Certificate, Private key, Root CA
   - Attach policy (see below)

**IoT Policy (allow all for testing):**
```json
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
```

#### 2. Launch EC2 Instance (~10 min)

**Specs:**
- Region: us-east-2 (Ohio) - same as IoT Core
- Instance type: t3.micro (Free tier eligible, ~$0.01/hr)
- AMI: Ubuntu 22.04 LTS
- Storage: 8GB (default)
- Security Group:
  - Inbound: Port 22 (SSH from your IP)
  - Inbound: Port 8883 (MQTT from anywhere OR your IP only)
  - Outbound: All

**Cost estimate:** $3-5/month if left running

#### 3. Configure EC2 Instance (~20 min)

**SSH to instance and install dependencies:**
```bash
ssh -i your-key.pem ubuntu@ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and required packages
sudo apt install -y python3-pip
pip3 install paho-mqtt

# Create directories
mkdir -p ~/mqtt-proxy/certs
mkdir -p ~/mqtt-proxy/logs
```

**Upload your AWS IoT certificates:**
```bash
# On your laptop:
scp -i your-key.pem certificate.pem.crt ubuntu@ec2-ip:~/mqtt-proxy/certs/
scp -i your-key.pem private.pem.key ubuntu@ec2-ip:~/mqtt-proxy/certs/
scp -i your-key.pem AmazonRootCA1.pem ubuntu@ec2-ip:~/mqtt-proxy/certs/
```

**Upload the proxy script:**
```bash
# We'll need to modify aws_local_proxy.py to:
# 1. Use AWS IoT Core MQTT instead of raw sockets
# 2. Subscribe to all topics (#)
# 3. Log messages
# 4. Republish to real endpoint
```

#### 4. Update Proxy Script

Need to create `aws_mqtt_proxy.py`:
- Accept MQTT connections on port 8883
- Use YOUR AWS IoT certificate
- Subscribe to all topics from real AWS IoT Core
- Log all messages
- Forward messages bidirectionally

#### 5. Enable DNS Spoofing on Laptop

```bash
# On laptop:
sudo mv ~/gcp2-dns-spoof.conf.backup /etc/NetworkManager/dnsmasq-shared.d/gcp2-dns-spoof.conf

# Edit to point to EC2 instance:
# address=/a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com/<EC2-PUBLIC-IP>

# Restart network
sudo nmcli connection down "Wired connection 1"
sudo nmcli connection up "Wired connection 1"
```

#### 6. Test Connection

1. Power cycle GCP2 device
2. Device resolves AWS hostname → gets EC2 IP
3. Connects to EC2 on port 8883
4. EC2 presents valid AWS certificate (device accepts!)
5. Proxy logs MQTT messages and forwards to real AWS

### Alternative: AWS IoT Core Rules (Simpler but Limited)

Instead of EC2 proxy, use IoT Rules to republish:

**Pros:**
- No EC2 instance needed
- Fully managed
- CloudWatch logging built-in

**Cons:**
- Can't forward to the ORIGINAL AWS IoT Core (don't have access)
- Would need to recreate the backend service
- More complex setup

**Verdict:** Use EC2 proxy for now since we just want to capture traffic.

## Files to Prepare Tomorrow

### 1. `aws_mqtt_proxy.py`
MQTT-based proxy using AWS IoT Core connection

### 2. `setup_ec2.sh`
Automated EC2 setup script

### 3. `DNS_SWITCH.md`
Quick reference for enabling/disabling DNS spoofing

## Estimated Timeline

| Task | Time | Cost |
|------|------|------|
| Set up AWS IoT Core | 15 min | Free |
| Launch EC2 | 10 min | ~$0.01/hr |
| Configure EC2 | 20 min | - |
| Write/test proxy script | 30-60 min | - |
| Test with device | 15 min | - |
| **Total** | **1.5-2 hours** | **~$5/month** |

## Quick Reference Commands

### Enable DNS Spoofing
```bash
sudo mv ~/gcp2-dns-spoof.conf.backup /etc/NetworkManager/dnsmasq-shared.d/gcp2-dns-spoof.conf
sudo nmcli connection down "Wired connection 1" && sudo nmcli connection up "Wired connection 1"
```

### Disable DNS Spoofing
```bash
sudo mv /etc/NetworkManager/dnsmasq-shared.d/gcp2-dns-spoof.conf ~/gcp2-dns-spoof.conf.backup
sudo nmcli connection down "Wired connection 1" && sudo nmcli connection up "Wired connection 1"
```

### Check Device Connection
```bash
sudo arp -i enxc8a362e1fa19 -n  # Should show 192.168.100.82
dig @192.168.100.1 a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com +short  # Shows where DNS points
```

## Questions to Resolve Tomorrow

1. ❓ Do we want to use raw sockets or MQTT library for the proxy?
2. ❓ Should we also capture encrypted traffic with tcpdump for comparison?
3. ❓ Do we need to handle device authentication/credentials?

## Current Network State

```
USB Ethernet: enxc8a362e1fa19
  ├─ IP: 192.168.100.1/24
  ├─ DHCP: Active (192.168.100.10-254)
  ├─ NAT: → WiFi (wlp0s20f3) ✅
  └─ DNS: Normal resolution (spoofing disabled)

GCP2 Device:
  ├─ IP: 192.168.100.82
  ├─ Status: Connected to real AWS IoT Core
  └─ DNS Spoofing: Ready to enable (config in ~/gcp2-dns-spoof.conf.backup)
```
