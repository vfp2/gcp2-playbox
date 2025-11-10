# Experiment 5: GCP2.NET RNG MQTT Traffic Interception

## Status: ✅ SUCCESS

Successfully intercepted MQTT traffic from GCP2.NET RNG device by extracting and registering the device's client certificate in our own AWS IoT Core account.

## Overview
Investigation into intercepting and analyzing MQTT traffic from GCP2.NET RNG devices connecting to AWS IoT Core. After exploring multiple approaches, we successfully captured the device's client certificate from packet analysis and registered it in our AWS account, allowing us to monitor all MQTT traffic.

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

### Solution Scripts
- **`extract_client_cert.py`** - Extract device's client certificate from PCAP ⭐
- **`monitor_mqtt.sh`** - Real-time monitoring of AWS IoT Core metrics ⭐
- `analyze_pcap.py` - Extract MQTT connection details from pcap files
- `decode_sni.py` - Parse and display TLS SNI (Server Name Indication) from pcap

### Alternative Approaches (Explored but not used)
- `mqtt_proxy.py` - EC2 proxy implementation (abandoned due to cert hostname mismatch)
- `aws_local_proxy.py` - Local AWS IoT proxy (same issue)
- `sni_proxy.py` - SNI-based transparent proxy (no decryption capability)
- `test_cert_validation.py` - Proof that device validates SSL certificates

### Documentation
- **`FINAL_SOLUTION.md`** - Complete technical documentation of the working solution ⭐
- `EC2_PROXY_SETUP.md` - EC2 proxy setup steps (alternative approach)
- `SELF_HOSTED_PROXY_SETUP.md` - Self-hosted proxy documentation
- `PROXY_OPTIONS.md` - Comparison of different proxy approaches
- `SNI_EXPLAINED.md` - Technical explanation of SNI in TLS handshakes
- `NEXT_STEPS.md` - Planning notes from earlier phases

## Final Solution

**See [FINAL_SOLUTION.md](FINAL_SOLUTION.md) for complete documentation.**

### Quick Summary

1. **Extracted device's client certificate** from packet capture (TLS 1.2 handshake)
2. **Registered certificate** in our AWS IoT Core account (us-east-2)
3. **DNS spoofed** original endpoint to resolve to our IoT endpoint IP
4. **Device connects successfully** - authenticates with its client cert
5. **All MQTT traffic visible** in AWS IoT Core console

### Current Status
- ✅ Device connecting (4+ successful connections observed)
- ✅ Device subscribing to topics
- ✅ CloudWatch logging enabled
- ⏳ Waiting for device to publish RNG data (likely buffering)

### Monitoring
```bash
# Real-time monitoring
./monitor_mqtt.sh

# AWS Console - MQTT Test Client
# Subscribe to topic: #
# https://us-east-2.console.aws.amazon.com/iot/home?region=us-east-2#/test
```

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
