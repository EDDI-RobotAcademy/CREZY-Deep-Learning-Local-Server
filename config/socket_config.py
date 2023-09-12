import socket
from decouple import config
# pip install python-decouple


class SocketConfig:
    def __init__(self):
        self.host = config('HOST')
        self.port = int(config('PORT'))
        self.server_socket = None

    def set_host(self, host):
        self.host = host

    def set_port(self, port):
        self.port = port

    def create_socket(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def get_server_socket(self):
        return self.server_socket

    def connect(self):
        if not self.server_socket:
            self.create_socket()

        try:
            self.server_socket.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
        except ConnectionRefusedError:
            print(f"Connection to {self.host}:{self.port} refused.")
        except Exception as e:
            print(f"Error while connecting: {str(e)}")

    def turn_on_non_blocking(self):
        self.server_socket.setblocking(False)

    def close(self):
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None


# 사용 예시
if __name__ == "__main__":
    config = SocketConfig()
    config.connect()
    config.close()
