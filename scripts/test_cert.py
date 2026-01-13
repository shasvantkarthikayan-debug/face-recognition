#!/usr/bin/env python
import os
from generate_cert import generate_selfsigned_cert

# Remove old files
for f in ['cert.pem', 'key.pem']:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed {f}")

# Generate new
generate_selfsigned_cert('cert.pem', 'key.pem')

# Verify
from OpenSSL import crypto
with open('cert.pem', 'rb') as f:
    cert_data = f.read()
    cert = crypto.load_certificate(crypto.FILETYPE_PEM, cert_data)
    
print(f"\nCertificate CN: {cert.get_subject().CN}")
print(f"Extensions: {cert.get_extension_count()}")

for i in range(cert.get_extension_count()):
    ext = cert.get_extension(i)
    name = ext.get_short_name().decode()
    print(f"  {name}: {ext}")
