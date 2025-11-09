# Experiment 5: TCP Dumping GCP2 RNGs

## Overview
Investigation into intercepting and analyzing MQTT traffic from GCP2.NET RNG devices connecting to AWS IoT Core.

## Discovered Configuration
- **Device**: GCP2.NET RNG
- **Protocol**: MQTT over TLS (port 8883)
- **Destination**: `a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com`
- **Service**: AWS IoT Core (us-east-2 region)
- **Certificate Validation**: ✅ ENABLED (device validates SSL certificates)

## Test Results

### Certificate Validation Test (2025-11-10)
**Setup:**
- USB Ethernet adapter with DHCP/DNS/NAT
- DNS spoofing to redirect AWS IoT hostname to local proxy
- Self-signed certificate presented to device

**Result:**
```
Error: [SSL: TLSV1_ALERT_UNKNOWN_CA] tlsv1 alert unknown ca
```

**Conclusion:**
- Device **DOES validate** SSL certificates
- Device **REJECTS** self-signed certificates
- Requires AWS-signed certificate for man-in-the-middle decryption

## Files

### Analysis Tools
- `analyze_pcap.py` - Extract MQTT connection details from pcap files
- `decode_sni.py` - Parse and display TLS SNI (Server Name Indication) from pcap

### Proxy Implementations
- `test_cert_validation.py` - Test if device validates SSL certificates
- `sni_proxy.py` - SNI-based transparent proxy (no decryption, metadata only)
- `aws_local_proxy.py` - AWS IoT proxy with TLS termination (requires valid AWS cert)

### Documentation
- `PROXY_OPTIONS.md` - Comparison of different proxy approaches
- `SNI_EXPLAINED.md` - Technical explanation of SNI in TLS handshakes

## Next Steps

### Option A: AWS EC2 Proxy (Recommended for full decryption)
1. Create AWS IoT Core endpoint in personal AWS account
2. Obtain valid AWS certificate for endpoint
3. Deploy proxy on EC2 instance
4. Use DNS spoofing to redirect device traffic
5. Log and forward decrypted MQTT messages

**Pros:**
- Full MQTT decryption (topics, payloads, RNG data)
- Device trusts certificate (valid AWS CA)
- Complete protocol analysis

**Cost:** ~$5/month for t3.micro EC2 instance

### Option B: SNI Proxy (Metadata only)
Use `sni_proxy.py` to capture:
- Connection timing and frequency
- Packet sizes
- Encrypted traffic patterns

**Pros:** Free, runs locally
**Cons:** Cannot see MQTT message content

## Network Setup Used

```
USB Ethernet (enxc8a362e1fa19)
  ├─ IP: 192.168.100.1/24
  ├─ DHCP: 192.168.100.10-254 (dnsmasq via NetworkManager)
  ├─ NAT: → WiFi (wlp0s20f3)
  └─ DNS: Custom dnsmasq config for spoofing

GCP2 Device
  ├─ IP: 192.168.100.82 (DHCP)
  └─ MAC: d4:d4:da:4a:94:93
```

## Key Findings

1. **SNI is visible**: Even without decryption, can see destination hostname in TLS Client Hello
2. **Certificate pinning**: Device validates certificates against Amazon CA
3. **Connection retries**: Device attempts reconnection ~23 times before giving up
4. **AWS IoT Core**: Uses Amazon Trust Services (ATS) certificates
