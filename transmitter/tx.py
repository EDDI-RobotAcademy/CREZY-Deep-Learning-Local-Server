import errno
import os
import queue
import socket
import threading
import time


class Transmitter:
    def __init__(self, transmit_queue):
        self.tx_thread = threading.Thread(target=self.transmit_response, name='TransmitterThread')
        self.thread_lwp = None
        self.pid = None
        self.running = True
        self.server_socket = None
        self.transmit_response_queue = transmit_queue

    def start_tx_thread(self, server_socket):
        self.server_socket = server_socket
        self.tx_thread.start()

    def transmit_response(self):
        while True:
            try:
                response = self.transmit_response_queue.get()

                self.server_socket.sendall(response.encode())

            except socket.error as e:
                if e.errno == errno.EWOULDBLOCK:
                    pass
                else:
                    print("Socket error:", e)
                    self.server_socket.close()
                    break

            except queue.Empty:
                time.sleep(1)

    def stop_tx_thread(self):
        pass
