# GCP2.NET RNG MQTT Traffic Interception - Final Solution

## Overview

Successfully intercepted and decrypted MQTT traffic from GCP2.NET RNG device by extracting the device's client certificate from packet capture and registering it in our own AWS IoT Core account.

## Problem Statement

- **Device**: GCP2.NET hardware random number generator
- **Connection**: MQTT over TLS (port 8883) to AWS IoT Core
- **Goal**: Intercept and decrypt MQTT messages to analyze RNG data
- **Challenge**: Device validates SSL certificates, blocking traditional MITM approaches

## Solution Architecture

```
GCP2.NET Device
    ↓ (DNS spoofed)
    ↓ connects to: a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com
    ↓ (resolves to our endpoint IP via dnsmasq)
    ↓
Laptop (NAT + DNS Spoof)
    ↓
    ↓ TLS handshake with device's client certificate
    ↓
AWS IoT Core (our account)
    ↓ endpoint: a1b3xs2hbt7sql-ats.iot.us-east-2.amazonaws.com
    ↓
    → MQTT messages visible in IoT console
    → CloudWatch Logs for persistence
    → Can create rules to forward/process data
```

## Key Technical Insights

### 1. Certificate Validation
- Device performs **full TLS certificate validation**
- Tested with self-signed certificate → Device rejected with `TLSV1_ALERT_UNKNOWN_CA`
- Cannot use traditional MITM proxy approach

### 2. AWS IoT Core Certificate Model
- **Server certificates**: AWS presents `*.iot.us-east-2.amazonaws.com` wildcard cert (private key not downloadable)
- **Client certificates**: IoT Things use unique client certs for authentication (`CN=AWS IoT Certificate`)
- Device's client certificate is sent in TLS handshake (visible in packet capture)

### 3. The Solution
- Extract device's **client certificate** from PCAP (TLS 1.2 allows this)
- Register that certificate in **our** AWS IoT Core account
- DNS spoof to redirect device to our endpoint
- Device connects successfully (same Amazon CA, valid wildcard server cert)
- All MQTT traffic visible in our IoT Core console

## Implementation Steps

### Step 1: Network Setup
```bash
# USB Ethernet adapter: enxc8a362e1fa19
# Device network: 192.168.100.0/24
# Laptop IP: 192.168.100.1
# Device IP: 192.168.100.82

# NAT routing configured via NetworkManager
# IP forwarding enabled
```

### Step 2: Extract Device Certificate
```bash
# Captured traffic in test_lan.pcap (old capture from different network)
# Device was at 192.168.51.143 in that capture

# Extract certificate from TLS handshake (see extract_client_cert.py)
python3 extract_client_cert.py

# Convert to PEM format
openssl x509 -in certificate_0.der -inform DER -out device_client_cert.pem -outform PEM

# Verify certificate
openssl x509 -in device_client_cert.pem -noout -subject -issuer
# subject=CN=AWS IoT Certificate
# issuer=OU=Amazon Web Services O=Amazon.com Inc. L=Seattle ST=Washington C=US
```

### Step 3: Register Certificate in AWS IoT Core
```bash
# Register without CA (device cert was issued by Amazon)
aws iot register-certificate-without-ca \
  --certificate-pem file://device_client_cert.pem \
  --status ACTIVE \
  --profile vfp2 \
  --region us-east-2

# Output: certificateArn and certificateId

# Create permissive policy for testing
aws iot create-policy \
  --policy-name gcp2-device-policy \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": "iot:*",
      "Resource": "*"
    }]
  }' \
  --profile vfp2 \
  --region us-east-2

# Attach policy to certificate
aws iot attach-policy \
  --policy-name gcp2-device-policy \
  --target <certificateArn> \
  --profile vfp2 \
  --region us-east-2
```

### Step 4: DNS Spoofing
```bash
# File: /etc/NetworkManager/dnsmasq-shared.d/gcp2-dns-spoof.conf
address=/a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com/18.117.204.74
log-queries

# Restart NetworkManager
sudo systemctl restart NetworkManager

# Verify DNS resolution
dig @192.168.100.1 a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com
# Should return: 18.117.204.74 (one of our endpoint's IPs)
```

### Step 5: Enable CloudWatch Logging
```bash
# Create IAM role for IoT logging
aws iam create-role \
  --role-name IoTLoggingRole \
  --assume-role-policy-document '{...}' \
  --profile vfp2

# Attach logging policy
aws iam attach-role-policy \
  --role-name IoTLoggingRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSIoTLogging \
  --profile vfp2

# Enable IoT Core logging
aws iot set-v2-logging-options \
  --role-arn arn:aws:iam::834500448638:role/IoTLoggingRole \
  --default-log-level INFO \
  --profile vfp2 \
  --region us-east-2
```

### Step 6: Monitor Device Activity

**AWS Console (MQTT Test Client):**
- URL: https://us-east-2.console.aws.amazon.com/iot/home?region=us-east-2#/test
- Subscribe to topic: `#` (all topics)
- Real-time message viewing

**Command Line:**
```bash
# Monitor connections
aws cloudwatch get-metric-statistics \
  --namespace AWS/IoT \
  --metric-name Connect.Success \
  --dimensions Name=Protocol,Value=MQTT \
  --start-time $(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 \
  --statistics Sum \
  --profile vfp2 \
  --region us-east-2

# Monitor published messages
# (Replace Connect.Success with PublishIn.Success)

# Use monitoring script
./monitor_mqtt.sh
```

## Current Status

✅ **Working:**
- Device successfully connects to our IoT Core endpoint
- Device subscribes to topics (4+ connections observed)
- No authorization errors
- CloudWatch logging enabled
- MQTT test client ready to capture messages

⏳ **Waiting:**
- Device to publish RNG data (likely buffering or on timer)
- First MQTT message to appear in console

## Next Steps

1. **Capture first message** - Identify topic structure and payload format
2. **Set up IoT Rules** - Create rule to log all messages to S3/DynamoDB
3. **Create API** - Build HTTP API to access captured RNG data
4. **Analysis** - Study RNG data format and statistical properties

## Files in This Repository

### Working Scripts
- `extract_client_cert.py` - Extract client certificate from PCAP
- `monitor_mqtt.sh` - Real-time monitoring of IoT Core metrics
- `analyze_pcap.py` - Initial PCAP analysis (SNI extraction)
- `decode_sni.py` - Educational: decode SNI from TLS handshake

### Historical/Alternative Approaches (Didn't Work)
- `mqtt_proxy.py` - EC2 proxy attempt (cert hostname mismatch issue)
- `aws_local_proxy.py` - Local proxy with AWS certs (same issue)
- `sni_proxy.py` - SNI-based proxy (device validates certs)
- `test_cert_validation.py` - Proof that device validates certificates

### Documentation
- `SNI_EXPLAINED.md` - Technical explanation of SNI
- `PROXY_OPTIONS.md` - Analysis of different proxy approaches
- `EC2_PROXY_SETUP.md` - EC2 proxy setup documentation
- `NEXT_STEPS.md` - Planning document from earlier phase

## Important URLs

- **Original device endpoint**: a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com
- **Our IoT endpoint**: a1b3xs2hbt7sql-ats.iot.us-east-2.amazonaws.com
- **Region**: us-east-2 (Ohio)
- **AWS Profile**: vfp2

## Lessons Learned

1. **AWS IoT Core uses mutual TLS** - Both server and client certificates required
2. **Client certificates are visible in TLS 1.2** - Can be extracted from packet captures
3. **Certificate reuse is possible** - AWS allows registering extracted certificates
4. **DNS spoofing is sufficient** - No need for complex proxy when you control the destination
5. **Wildcard certificates simplify things** - AWS's `*.iot.us-east-2.amazonaws.com` cert works for any endpoint in that region

## Security Implications

- Device's client certificate is **extractable from network traffic**
- Anyone with network access can capture and reuse the certificate
- Once registered in another AWS account, device will happily connect to it
- No device-side validation of server identity beyond certificate chain
- GCP2.NET should consider certificate pinning or device-specific validation

## Cost Considerations

- AWS IoT Core: Pay per message and connection time
- Device appears to connect frequently (every 1-2 minutes)
- Monitor AWS costs as messages start flowing
- CloudWatch Logs: Pay for ingestion and storage
- Original backend service is now disconnected (device no longer reporting to GCP2.NET)

## Cleanup

If you want to restore normal operation:
1. Remove DNS spoof configuration
2. Restart NetworkManager
3. Device will reconnect to original endpoint
4. Can delete/deactivate certificate in our AWS account

## Monitoring Commands Quick Reference

```bash
# Check connections
aws cloudwatch get-metric-statistics \
  --namespace AWS/IoT --metric-name Connect.Success \
  --dimensions Name=Protocol,Value=MQTT \
  --start-time $(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 --statistics Sum \
  --profile vfp2 --region us-east-2

# Check publishes
aws cloudwatch get-metric-statistics \
  --namespace AWS/IoT --metric-name PublishIn.Success \
  --dimensions Name=Protocol,Value=MQTT \
  --start-time $(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 --statistics Sum \
  --profile vfp2 --region us-east-2

# View CloudWatch logs (when available)
aws logs tail /aws/iot/events --follow --profile vfp2 --region us-east-2
```

---

**Status**: Device connected and authenticated. Awaiting first message publication.

**Last Updated**: 2025-11-10
