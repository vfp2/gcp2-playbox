# SNI (Server Name Indication) Explained

## The TLS Handshake Flow

```
GCP2 Device                                    AWS IoT Core
192.168.51.143                                 18.216.115.227
    |                                                |
    |  1. TCP SYN                                    |
    |----------------------------------------------->|
    |  2. TCP SYN-ACK                                |
    |<-----------------------------------------------|
    |  3. TCP ACK                                    |
    |----------------------------------------------->|
    |                                                |
    |  4. TLS Client Hello (UNENCRYPTED!)            |
    |     - TLS Version: 1.2                         |
    |     - Random (32 bytes)                        |
    |     - Cipher Suites                            |
    |     - Extensions:                              |
    |       ★ SNI: a1vfh7jker84ic-ats.iot...         |  ← VISIBLE!
    |       - Supported Groups                       |
    |       - Signature Algorithms                   |
    |----------------------------------------------->|
    |                                                |
    |                                    [Server reads SNI]
    |                                    [Selects correct cert]
    |                                                |
    |  5. TLS Server Hello (UNENCRYPTED!)            |
    |     - Selected cipher suite                    |
    |     - Certificate for *.iot.us-east-2...       |  ← Based on SNI!
    |     - Server random                            |
    |<-----------------------------------------------|
    |                                                |
    | [Client verifies certificate]                  |
    | [Generates session keys]                       |
    |                                                |
    |  6. Client Key Exchange (ENCRYPTED)            |
    |----------------------------------------------->|
    |                                                |
    | ═══════════════════════════════════════════════|
    |     ALL SUBSEQUENT TRAFFIC IS ENCRYPTED        |
    | ═══════════════════════════════════════════════|
    |                                                |
    |  7. MQTT CONNECT (encrypted)                   |
    |----------------------------------------------->|
    |  8. MQTT CONNACK (encrypted)                   |
    |<-----------------------------------------------|
```

## Why SNI is Sent in Plaintext

**Chicken and Egg Problem:**

1. Server needs to present the **correct certificate** to the client
2. But TLS encryption requires **exchanging keys first**
3. Key exchange requires **verifying the certificate**
4. Certificate depends on **which hostname** you want

**Solution:** Send hostname (SNI) **before** encryption starts!

## What Can Be Seen Without Decryption

### ✅ Visible (from your pcap)
- Source/Destination IPs: `192.168.51.143` → `18.216.115.227`
- Port: `8883`
- SNI Hostname: `a1vfh7jker84ic-ats.iot.us-east-2.amazonaws.com`
- TLS version: `1.2`
- Cipher suites offered
- Certificate chain (public info)
- Packet sizes and timing
- Connection duration

### ❌ Not Visible (encrypted)
- MQTT topics
- MQTT payloads (the RNG data!)
- Authentication credentials
- Message content
- Application-layer protocol details

## SNI and Your Proxy Options

### Option 1: SNI Proxy (No Decryption)
```python
# Read SNI from Client Hello
sni = extract_sni(client_data)  # "a1vfh7jker84ic-ats.iot..."

# Forward encrypted traffic to that destination
connect_to(sni, port=8883)
```
**Can see:** SNI, timing, packet sizes
**Cannot see:** MQTT content

### Option 2: TLS Terminating Proxy (With Decryption)
```
Device → [TLS with your cert] → Proxy → [TLS with AWS cert] → AWS IoT
         Client trusts you              You connect as client
```
**Can see:** Everything (full MQTT messages)
**Requires:** Device to trust your certificate OR use AWS cert

## Real-World Impact

### Privacy Concern
Even though HTTPS/TLS encrypts content, **SNI leaks which sites you visit:**

```
Visible in SNI:
  - banking.example.com
  - medical-records.hospital.com
  - job-search.linkedin.com
```

An ISP/government can see you visited these sites (but not what you did there).

### Solution: Encrypted Client Hello (ECH)
Modern fix (still being deployed):
- Encrypts SNI field
- Uses DNS records to get encryption key
- Requires both client and server support

## How I Extracted SNI From Your Pcap

```python
# From packet #9 in test_lan.pcap
payload = bytes(packet[Raw].load)

# TLS Record: 0x16 = Handshake
# Handshake Type: 0x01 = Client Hello
# Extension Type: 0x0000 = SNI

# At byte position 43+, found:
0x00 0x00                    # Extension type: SNI
0x00 0x33                    # Length: 51 bytes
0x00 0x31                    # List length: 49 bytes
0x00                         # Name type: hostname
0x00 0x2e                    # Name length: 46 bytes
61 31 76 66 68 37 6a...      # ASCII: "a1vfh7jker84ic-ats.iot..."
```

## For Your Use Case

**Good news:** You can see the destination hostname (`a1vfh7jker84ic-ats.iot...`) even without decrypting!

**Bad news:** You need to decrypt TLS to see:
- What topics the device publishes to
- What the actual RNG data looks like
- How often it sends data
- Message format and structure

This is why your AWS-based proxy idea is so clever - you get a valid AWS certificate, device trusts it, you can decrypt and log everything!
