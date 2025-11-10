#!/usr/bin/env python3
"""
Extract client certificate from TLS handshake in pcap
"""
import sys
from scapy.all import rdpcap, TLS, TLSCertificate
from scapy.layers.tls.handshake import TLSCertificateList

try:
    packets = rdpcap('test_lan.pcap')
    print(f"Loaded {len(packets)} packets")

    for i, pkt in enumerate(packets):
        if pkt.haslayer(TLS):
            print(f"\nPacket {i}: TLS packet found")

            # Check for certificate messages
            if pkt.haslayer(TLSCertificate):
                print(f"  -> Certificate message found!")
                cert_layer = pkt[TLSCertificate]
                print(f"  -> Certificate data: {cert_layer}")

            # Try to find any certificate list
            if pkt.haslayer(TLSCertificateList):
                print(f"  -> Certificate list found!")
                cert_list = pkt[TLSCertificateList]
                print(f"  -> Number of certs: {len(cert_list.certificates)}")

                for j, cert in enumerate(cert_list.certificates):
                    print(f"     Cert {j}: {len(cert)} bytes")
                    with open(f'device_cert_{i}_{j}.der', 'wb') as f:
                        f.write(cert)
                    print(f"     Saved to device_cert_{i}_{j}.der")

except ImportError:
    print("Scapy not installed. Trying alternative method...")
    import dpkt

    with open('test_lan.pcap', 'rb') as f:
        pcap = dpkt.pcap.Reader(f)

        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if isinstance(eth.data, dpkt.ip.IP):
                    ip = eth.data
                    if isinstance(ip.data, dpkt.tcp.TCP):
                        tcp = ip.data
                        if len(tcp.data) > 0:
                            # Check if it looks like TLS
                            if tcp.data[0] == 0x16:  # TLS Handshake
                                print(f"TLS handshake at {ts}")
                                # Very basic parsing
                                if len(tcp.data) > 5:
                                    msg_type = tcp.data[5]
                                    if msg_type == 0x0b:  # Certificate
                                        print("  -> Certificate message!")
            except:
                pass

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
