import queue


class ReceiveQueue:
    def __init__(self):
        self.receive_queue = queue.Queue()

    def put(self, data):
        self.receive_queue.put(data)

    def get(self):
        return self.receive_queue.get()


if __name__ == "__main__":
    receiver_queue = ReceiveQueue()

    receiver_queue.put((333, "화가 난다"))
    print("receiver response queue data: ", receiver_queue.get())
