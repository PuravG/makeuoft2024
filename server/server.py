import socketio

# Create a Socket.IO server
sio = socketio.Server()

# Define event handler for new connections
@sio.event
def connect(sid, environ):
    print('connect ', sid)
    sio.emit('message', 'Hello, client!', room=sid)

# Define event handler for messages received from clients
@sio.event
def message(sid, data):
    print('message ', data)
    # Echo the message back to the client
    sio.emit('message', data, room=sid)

# Define event handler for disconnections
@sio.event
def disconnect(sid):
    print('disconnect ', sid)

# Wrap the Socket.IO server with a WSGI application
app = socketio.WSGIApp(sio)

if __name__ == '__main__':
    import eventlet
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 3000)), app)
