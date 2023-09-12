import logging
import time

from config.socket_config import SocketConfig
from task_manager.manager import Manager


class SocketClient:
    def __init__(self):
        self.socket_config = SocketConfig()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.manager = Manager()

    def run(self):
        try:
            self.socket_config.connect()

            server_socket = self.socket_config.get_server_socket()

            self.socket_config.turn_on_non_blocking()
            self.manager.start_all_threads(server_socket)

            while True:
                time.sleep(1)

        except Exception as e:
            self.logger.info(e)


if __name__ == '__main__':
    app = SocketClient()
    app.run()
