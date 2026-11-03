import sys

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

        # 关键：每次写完都刷新
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        pass

    def end(self):
        self.log.close()


