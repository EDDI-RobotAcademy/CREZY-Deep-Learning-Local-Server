import queue


class CommandQueue:
    def __init__(self):
        self.command_queue = queue.Queue()

    def put(self, data):
        self.command_queue.put(data)

    def get(self):
        return self.command_queue.get()


if __name__ == "__main__":
    command_queue = CommandQueue()

    command_queue.put("화가 난다")
    print("receiver response queue data: ", command_queue.get())
