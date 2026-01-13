"""Generate self-signed SSL certificate for HTTPS with SAN entries"""
from OpenSSL import crypto
import os

def generate_selfsigned_cert(cert_file, key_file):
    # Create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)

    # Create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = "US"
    cert.get_subject().ST = "State"
    cert.get_subject().L = "City"
    cert.get_subject().O = "Organization"
    cert.get_subject().OU = "Organizational Unit"
    cert.get_subject().CN = "localhost"
    
    # Add Subject Alternative Name (SAN) extension
    # This is critical for modern browsers to accept the certificate
    san_list = [
        b"DNS:localhost",
        b"DNS:127.0.0.1",
        b"IP:127.0.0.1"
    ]
    san_extension = crypto.X509Extension(
        b"subjectAltName",
        False,
        b", ".join(san_list)
    )
    
    cert.add_extensions([san_extension])
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for one year
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')

    # Write certificate and private key to files
    with open(cert_file, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open(key_file, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    print(f"[OK] Generated certificate: {cert_file}")
    print(f"[OK] Generated private key: {key_file}")
    print(f"[OK] Added SAN entries: localhost, 127.0.0.1")

if __name__ == "__main__":
    cert_file = "cert.pem"
    key_file = "key.pem"
    
    # Remove old certificates if they exist
    if os.path.exists(cert_file):
        os.remove(cert_file)
        print(f"[DEL] Removed old {cert_file}")
    if os.path.exists(key_file):
        os.remove(key_file)
        print(f"[DEL] Removed old {key_file}")
    
    generate_selfsigned_cert(cert_file, key_file)
