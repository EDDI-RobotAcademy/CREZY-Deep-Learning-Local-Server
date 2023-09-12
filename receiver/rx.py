import errno
import socket
import threading
import time

from datetime import datetime as dt


class Receiver:
    def __init__(self, receive_queue):
        self.rx_thread = threading.Thread(target=self.receive_command,
                                          name='ReceiverThread')
        self.receiver_command_data_queue = receive_queue
        self.server_socket = None

    def start_rx_thread(self, server_socket):
        self.server_socket = server_socket
        self.rx_thread.start()

    def receive_command(self):
        while True:
            try:
                data = self.server_socket.recv(1024)
                if not data:
                    print("Server disconnected!")
                    self.server_socket.close()
                    break

                split_data = data.decode().split(',')
                self.receiver_command_data_queue.put(split_data)

            except socket.error as e:
                if e.errno == errno.EWOULDBLOCK:
                    pass

            finally:
                time.sleep(0.5)

    def stop_rx_thread(self):
        pass
