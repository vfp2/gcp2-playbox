# MQTT Proxy Options for GCP2.NET Device

## Discovered Configuration
- **Device**: GCP2.NET RNG at 192.168.51.143
- **Destination**: a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com
- **Protocol**: MQTT over TLS (port 8883)
- **Service**: AWS IoT Core (us-east-2)

---

## Option 1: Local SNI Proxy (NO Decryption)

**Location**: Runs locally on your network
**Certificate**: Not needed - traffic stays encrypted
**Decryption**: ❌ No - only sees encrypted data

### Setup
```bash
# 1. Run the SNI proxy locally
python3 sni_proxy.py

# 2. DNS spoof (on router/Pi-hole)
192.168.x.x  a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com
```

### What You Get
- ✅ Connection timing and metadata
- ✅ Packet sizes and frequency
- ✅ SNI hostname extraction
- ✅ Full encrypted traffic capture
- ❌ Cannot see MQTT topics/payloads

### Best For
- Quick setup, no AWS account needed
- Statistical analysis of traffic patterns
- When you just need timing/volume data

---

## Option 2: AWS EC2 Proxy (Full Decryption)

**Location**: Runs on AWS EC2 instance
**Certificate**: Valid AWS IoT Core cert (from your AWS account)
**Decryption**: ✅ Yes - sees plaintext MQTT

### Setup
```bash
# 1. Create AWS IoT Core endpoint in YOUR account
# 2. Download your endpoint's certificate
# 3. Launch EC2 instance in us-east-2
# 4. Run aws_local_proxy.py on EC2
# 5. DNS spoof to point to your EC2 instance
#    OR create IoT Rule to forward traffic

# 6. (Optional) SSH tunnel logs back to local machine
ssh -R 9999:localhost:9999 ec2-user@your-instance
```

### What You Get
- ✅ Full MQTT message content (topics, payloads)
- ✅ Device authentication details
- ✅ Complete protocol analysis
- ✅ Valid AWS certificate (device trusts it)

### Cost
- ~$3-10/month for t3.micro EC2 instance
- Minimal data transfer costs
- Free tier eligible

### Best For
- Complete traffic analysis
- Understanding data format
- When you want plaintext MQTT logs

---

## Option 3: Local Proxy with Self-Signed Cert

**Location**: Runs locally
**Certificate**: Self-signed (requires device to trust it OR disable verification)
**Decryption**: ✅ Yes (if device accepts cert)

### Setup
```bash
# 1. Generate certificate
openssl req -new -x509 -days 365 -nodes \
  -out proxy.crt -keyout proxy.key \
  -subj "/CN=a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com"

# 2. Run local TLS-terminating proxy
# (Would need to modify aws_local_proxy.py to use self-signed cert)

# 3. DNS spoof to localhost
127.0.0.1  a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com
```

### Challenges
- ⚠️ Device likely validates AWS certificate
- ⚠️ Cannot install custom CA on GCP2.NET device
- ⚠️ May not work unless device has `verify=false`

### Testing If This Works
```bash
# Test if device validates certificates
# Block original AWS IPs with firewall, point DNS to local proxy
# If connection fails = certificate validation enabled
```

---

## Recommended Approach: Hybrid (2a)

**Best of both worlds:**

1. **Run proxy on AWS EC2** with valid cert → decrypts traffic
2. **Stream logs to local machine** via SSH tunnel or S3
3. **Optional: Also run local SNI proxy** for backup encrypted capture

### Architecture
```
Device → DNS Spoof → AWS EC2 Proxy → Real AWS IoT Core
                          ↓
                     SSH Tunnel
                          ↓
                    Local Machine
                    (stores logs)
```

### Implementation
```bash
# On AWS EC2
python3 aws_local_proxy.py  # Logs to mqtt_traffic.log

# On local machine
ssh -N -f ec2-user@ec2-ip -L 2222:localhost:22
scp -r ec2-user@localhost:mqtt_traffic.log ./logs/
# Or use rsync in cron for continuous sync
```

---

## Quick Decision Guide

**Want it working in 5 minutes?**
→ Use **Option 1** (SNI Proxy) - metadata only

**Need to see actual MQTT messages?**
→ Use **Option 2** (AWS EC2 Proxy) - full decryption

**No AWS account / want free solution?**
→ Try **Option 3** first (might not work due to cert validation)
→ Fallback to **Option 1** (SNI Proxy)

**Maximum data collection?**
→ Use **Hybrid**: AWS proxy + local SNI proxy + tcpdump on both
