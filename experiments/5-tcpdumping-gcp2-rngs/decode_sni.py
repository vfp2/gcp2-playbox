#!/usr/bin/env python3
"""
Decode and visualize SNI from TLS Client Hello in pcap
"""
from scapy.all import rdpcap, TCP, Raw, IP
import struct

def parse_tls_client_hello(payload):
    """Parse TLS Client Hello and extract SNI"""

    if len(payload) < 43:
        return None

    # TLS Record Layer
    content_type = payload[0]
    version_major = payload[1]
    version_minor = payload[2]
    record_length = struct.unpack('!H', payload[3:5])[0]

    print("\n=== TLS Record Layer ===")
    print(f"Content Type: 0x{content_type:02x} (0x16 = Handshake)")
    print(f"Version: {version_major}.{version_minor} (3.1 = TLS 1.0, 3.3 = TLS 1.2)")
    print(f"Record Length: {record_length} bytes")

    # Handshake Protocol
    handshake_type = payload[5]
    handshake_length = struct.unpack('!I', b'\x00' + payload[6:9])[0]

    print("\n=== Handshake Protocol ===")
    print(f"Handshake Type: 0x{handshake_type:02x} (0x01 = Client Hello)")
    print(f"Handshake Length: {handshake_length} bytes")

    if handshake_type != 0x01:
        return None

    # Client Hello
    client_version = f"{payload[9]}.{payload[10]}"
    print(f"Client Version: {client_version}")

    # Skip random (32 bytes) and session ID
    pos = 11 + 32  # After version + random
    session_id_len = payload[pos]
    pos += 1 + session_id_len

    # Skip cipher suites
    cipher_suites_len = struct.unpack('!H', payload[pos:pos+2])[0]
    print(f"Cipher Suites Length: {cipher_suites_len} bytes")
    pos += 2 + cipher_suites_len

    # Skip compression methods
    compression_len = payload[pos]
    pos += 1 + compression_len

    # Extensions
    if pos + 2 > len(payload):
        return None

    extensions_len = struct.unpack('!H', payload[pos:pos+2])[0]
    pos += 2

    print(f"\n=== Extensions (total {extensions_len} bytes) ===")

    end = pos + extensions_len
    extension_data = {}

    while pos + 4 <= end:
        ext_type = struct.unpack('!H', payload[pos:pos+2])[0]
        ext_len = struct.unpack('!H', payload[pos+2:pos+4])[0]
        pos += 4

        ext_types = {
            0: 'server_name (SNI)',
            1: 'max_fragment_length',
            5: 'status_request',
            10: 'supported_groups',
            11: 'ec_point_formats',
            13: 'signature_algorithms',
            16: 'application_layer_protocol_negotiation',
            23: 'extended_master_secret',
            35: 'session_ticket',
            43: 'supported_versions',
            45: 'psk_key_exchange_modes',
            51: 'key_share'
        }

        ext_name = ext_types.get(ext_type, f'unknown({ext_type})')
        print(f"  Extension {ext_type}: {ext_name} ({ext_len} bytes)")

        if ext_type == 0 and ext_len > 0:  # SNI
            # SNI structure: server_name_list_length (2) + name_type (1) + name_length (2) + name
            sni_list_len = struct.unpack('!H', payload[pos:pos+2])[0]
            name_type = payload[pos+2]  # 0 = hostname
            name_len = struct.unpack('!H', payload[pos+3:pos+5])[0]
            hostname = payload[pos+5:pos+5+name_len].decode('utf-8', errors='ignore')

            print(f"    → SNI Hostname: {hostname}")
            print(f"    → Name Type: {name_type} (0 = host_name)")
            print(f"    → Name Length: {name_len} bytes")

            extension_data['sni'] = hostname

        pos += ext_len

    return extension_data

def main():
    print("="*70)
    print("SNI (Server Name Indication) Analysis")
    print("="*70)

    pkts = rdpcap('test_lan.pcap')

    for i, pkt in enumerate(pkts):
        if pkt.haslayer(TCP) and pkt.haslayer(Raw):
            payload = bytes(pkt[Raw].load)

            # Check for TLS Client Hello
            if len(payload) > 5 and payload[0] == 0x16 and payload[5] == 0x01:
                print(f"\n{'='*70}")
                print(f"PACKET #{i} - TLS Client Hello")
                print(f"{'='*70}")

                if pkt.haslayer(IP):
                    print(f"Source: {pkt[IP].src}:{pkt[TCP].sport}")
                    print(f"Destination: {pkt[IP].dst}:{pkt[TCP].dport}")

                result = parse_tls_client_hello(payload)

                if result and 'sni' in result:
                    print(f"\n{'*'*70}")
                    print(f"*** SNI EXTRACTED: {result['sni']} ***")
                    print(f"{'*'*70}")

                    print("\n=== Why SNI is Unencrypted ===")
                    print("• SNI is sent BEFORE TLS encryption starts")
                    print("• Server needs to know which certificate to present")
                    print("• This is why we can see the hostname in plaintext")
                    print("• Anyone monitoring traffic can see which sites you visit")
                    print("\n=== Privacy Note ===")
                    print("• Encrypted SNI (ESNI/ECH) is being developed to fix this")
                    print("• Currently, SNI leaks the destination hostname")

                break  # Only show first Client Hello

if __name__ == "__main__":
    main()
