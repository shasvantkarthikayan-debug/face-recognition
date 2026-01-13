"""
Production server runner with HTTPS support
Architecture:
- Waitress serves HTTP on 127.0.0.1:5001 (backend)
- SSL proxy listens on 127.0.0.1:5000 (frontend with HTTPS)
- Browser connects to https://127.0.0.1:5000
"""
import os
import ssl
import socket
import threading
from waitress import serve
from app import app

def generate_self_signed_cert(cert_file='cert.pem', key_file='key.pem'):
    """Generate a self-signed certificate with SAN for localhost + 127.0.0.1."""
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"✓ Using existing certificates: {cert_file}, {key_file}")
        return

    print("🔐 Generating self-signed certificate (with SAN)...")

    # Use cryptography to generate a cert browsers will accept (after you bypass trust warning)
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime
    import ipaddress

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FacePass"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    san = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ])

    now = datetime.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )

    # Write files
    with open(cert_file, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    with open(key_file, 'wb') as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    print(f"✓ Generated certificates: {cert_file}, {key_file}")

def ssl_proxy(cert_file, key_file, frontend_port=5000, backend_port=5001):
    """
    Simple SSL proxy that forwards HTTPS requests to HTTP backend
    """
    print(f"🔒 Starting SSL proxy on port {frontend_port}")
    print(f"   Forwarding to backend on port {backend_port}")

    # Create SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)

    # Create listening socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', frontend_port))
        sock.listen(5)

        print(f"✓ SSL proxy listening on https://127.0.0.1:{frontend_port}")

        with context.wrap_socket(sock, server_side=True) as ssock:
            while True:
                try:
                    client_sock, addr = ssock.accept()
                    # Handle connection in a new thread
                    thread = threading.Thread(
                        target=handle_client,
                        args=(client_sock, backend_port),
                        daemon=True
                    )
                    thread.start()
                except KeyboardInterrupt:
                    print("\n⏹ Shutting down SSL proxy...")
                    break
                except Exception as e:
                    print(f"⚠️ SSL proxy error: {e}")

def handle_client(client_sock, backend_port):
    """Forward client request to backend and return response"""
    try:
        # Receive request from client
        request = client_sock.recv(8192)
        if not request:
            return

        # Forward to backend
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as backend_sock:
            backend_sock.connect(('127.0.0.1', backend_port))
            backend_sock.sendall(request)

            # Get response and forward to client
            while True:
                data = backend_sock.recv(8192)
                if not data:
                    break
                client_sock.sendall(data)
    except Exception as e:
        print(f"⚠️ Client handler error: {e}")
    finally:
        client_sock.close()

if __name__ == '__main__':
    print("="*60)
    print("🚀 FacePass Production Server")
    print("="*60)

    CERT_FILE = 'cert.pem'
    KEY_FILE = 'key.pem'
    FRONTEND_PORT = 5000
    BACKEND_PORT = 5001

    # Generate certificate if needed
    try:
        generate_self_signed_cert(CERT_FILE, KEY_FILE)
    except Exception as e:
        print(f"\n❌ Certificate generation failed: {e}")
        print("\n💡 Running without HTTPS on port 5000...")
        # Fallback: Run without HTTPS
        print("\n🌐 Starting server on http://127.0.0.1:5000")
        serve(app, host='127.0.0.1', port=5000, threads=4)
        exit(0)

    # Start backend server in a thread
    print(f"\n🔧 Starting backend server on port {BACKEND_PORT}...")
    backend_thread = threading.Thread(
        target=lambda: serve(app, host='127.0.0.1', port=BACKEND_PORT, threads=4),
        daemon=True
    )
    backend_thread.start()

    # Wait a moment for backend to start
    import time
    time.sleep(1)
    print(f"✓ Backend running on http://127.0.0.1:{BACKEND_PORT}")

    # Start SSL proxy
    print("\n" + "="*60)
    print(f"✅ Server ready!")
    print(f"   HTTPS: https://127.0.0.1:{FRONTEND_PORT}")
    print(f"   HTTP:  http://127.0.0.1:{BACKEND_PORT} (backend)")
    print("\n⚠️  Browser will show security warning (self-signed cert)")
    print("   Click 'Advanced' → 'Proceed to localhost'")
    print("="*60 + "\n")

    try:
        ssl_proxy(CERT_FILE, KEY_FILE, FRONTEND_PORT, BACKEND_PORT)
    except KeyboardInterrupt:
        print("\n\n⏹ Shutting down server...")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()