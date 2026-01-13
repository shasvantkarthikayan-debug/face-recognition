"""Run Flask app with HTTPS"""
from app import app
import os
import ssl

if __name__ == '__main__':
    cert_file = 'cert.pem'
    key_file = 'key.pem'
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"✓ Starting HTTPS server on https://127.0.0.1:5000")
        print(f"✓ Using certificate: {cert_file}")
        print(f"✓ Camera should work!")
        print(f"⚠️  You'll see a security warning - click 'Advanced' then 'Proceed'")
        print(f"🔄 Server is running... Press Ctrl+C to stop")
        
        # Create SSL context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file, key_file)
        
        # Run with SSL and threaded mode
        app.run(host='127.0.0.1', port=5000, 
                ssl_context=context,
                debug=False, use_reloader=False, threaded=True)
    else:
        print("ERROR: SSL certificate files not found!")
        print("Run: python generate_cert.py")
