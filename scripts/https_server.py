"""HTTPS server using http.server with Flask app"""
from http.server import HTTPServer, BaseHTTPRequestHandler
from werkzeug.serving import WSGIRequestHandler
import ssl
from app import app

class FlaskHandler(WSGIRequestHandler):
    pass

if __name__ == '__main__':
    port = 5000
    server_address = ('127.0.0.1', port)
    
    # Create HTTPS server
    httpd = HTTPServer(server_address, FlaskHandler)
    httpd.set_app(app)
    
    # Wrap with SSL
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('cert.pem', 'key.pem')
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    
    print(f"✅ HTTPS Server running on https://127.0.0.1:{port}")
    print(f"⚠️  You'll see a security warning - click 'Advanced' then 'Proceed to 127.0.0.1'")
    print(f"🔄 Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.shutdown()
