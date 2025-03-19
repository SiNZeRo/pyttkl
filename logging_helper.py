import datetime
import os
import time
import multiprocessing
import logging

import datetime
import os
import time
import multiprocessing

def log_message(log_file, message, retries=5, delay=1):
    import portalocker
    whoami_process = multiprocessing.current_process().name
    user_name = os.getenv('USER')
    for _ in range(retries):
        try:
            with open(log_file, 'a') as file:
                portalocker.lock(file, portalocker.LOCK_EX)
                try:
                    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
                    payload = f'{timestamp} {user_name} {whoami_process}\t|{message}'
                    file.write(payload + '\n')
                finally:
                    portalocker.unlock(file)
            return
        except portalocker.exceptions.LockException:
            time.sleep(delay)
    raise Exception("Could not acquire file lock after several retries")

class FileLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.queue = multiprocessing.Queue()
        self.worker = multiprocessing.Process(target=self._working_thread, args=(self.queue,))
        self.worker.start()

    def __del__(self):
        self.queue.put((None, None))
        self.worker.join()

    def log(self, message):
        log_message(self.log_file, message)

    def log_async(self, message):
        self.queue.put(message)

    def log_sync(self, message):
        self.log((message, datetime.datetime.utcnow()))

    def _working_thread(self, queue):
        while True:
            # pull all messages from the queue
            messages = []
            while not queue.empty():
                msg = queue.get_nowait()
                if msg is None:
                    break


            # log all messages
