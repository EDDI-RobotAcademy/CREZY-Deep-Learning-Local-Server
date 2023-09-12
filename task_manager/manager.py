import time

from command_analyzer.analyzer import CommandAnalyzer
from command_executor.executor import CommandExecutor
from ipc.command.message_queue import CommandQueue
from ipc.receiver.message_queue import ReceiveQueue
from ipc.transmitter.message_queue import TransmitQueue
from receiver.rx import Receiver
from transmitter.tx import Transmitter

import subprocess
import platform
import os


class Manager:
    def __init__(self):
        receive_queue = ReceiveQueue()
        command_queue = CommandQueue()
        transmit_queue = TransmitQueue()

        self.receiver = Receiver(receive_queue)
        self.transmitter = Transmitter(transmit_queue)
        self.command_analyzer = CommandAnalyzer(receive_queue, command_queue)
        self.command_executor = CommandExecutor(command_queue, transmit_queue)

    def start_all_threads(self, server_socket):
        self.receiver.start_rx_thread(server_socket)
        self.command_analyzer.start_analyzer_thread()
        self.command_executor.start_executor_thread()
        self.transmitter.start_tx_thread(server_socket)

    def stop_all_threads(self):
        self.receiver.stop_rx_thread()
        self.transmitter.stop_tx_thread()
        self.command_analyzer.stop_analyzer_thread()
        self.command_executor.stop_executor_thread()
