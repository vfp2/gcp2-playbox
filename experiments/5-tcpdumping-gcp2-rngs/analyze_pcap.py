#!/usr/bin/env python3
from scapy.all import rdpcap, IP, TCP
from collections import Counter

print("Analyzing test_lan.pcap...")
pkts = rdpcap('test_lan.pcap')

# Find all TCP connections on port 8883
mqtt_endpoints = []
for pkt in pkts:
    if pkt.haslayer(IP) and pkt.haslayer(TCP):
        tcp = pkt[TCP]
        ip = pkt[IP]
        if tcp.dport == 8883:
            mqtt_endpoints.append((ip.src, ip.dst, tcp.sport, tcp.dport))
        elif tcp.sport == 8883:
            mqtt_endpoints.append((ip.dst, ip.src, tcp.dport, tcp.sport))

if mqtt_endpoints:
    print("\n=== MQTT Connections (port 8883) ===")
    # Get unique connections
    connections = set([(src, dst) for src, dst, sp, dp in mqtt_endpoints])
    for src, dst in connections:
        print(f"Client: {src} -> Server: {dst}:8883")

    # Get most common IPs
    src_ips = [src for src, dst, sp, dp in mqtt_endpoints]
    dst_ips = [dst for src, dst, sp, dp in mqtt_endpoints]

    print("\n=== Traffic Summary ===")
    print(f"Total packets: {len(pkts)}")
    print(f"MQTT packets: {len(mqtt_endpoints)}")
    print(f"\nClient IP(s): {set(src_ips)}")
    print(f"Server IP(s): {set(dst_ips)}")

    # Try to extract SNI from TLS Client Hello
    print("\n=== Looking for TLS SNI (Server Name) ===")
    for pkt in pkts:
        if pkt.haslayer(TCP) and len(pkt[TCP].payload) > 0:
            payload = bytes(pkt[TCP].payload)
            # Look for TLS Client Hello (0x16 0x03)
            if len(payload) > 5 and payload[0] == 0x16 and payload[1] == 0x03:
                # Try to find SNI extension (very simplified)
                if b'aws' in payload.lower() or b'gcp' in payload.lower():
                    # Try to extract readable hostname
                    import re
                    hostnames = re.findall(b'([a-zA-Z0-9\-\.]+\.(?:amazonaws\.com|iot\.us-\w+-\d+\.amazonaws\.com))', payload)
                    if hostnames:
                        print(f"Found hostname: {hostnames[0].decode()}")
else:
    print("No MQTT traffic on port 8883 found!")
    print("\n=== All unique ports ===")
    ports = set()
    for pkt in pkts:
        if pkt.haslayer(TCP):
            ports.add(pkt[TCP].sport)
            ports.add(pkt[TCP].dport)
    print(sorted(ports))
