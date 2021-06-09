from abc import ABC, abstractmethod
import logging
import time
import prettytable as pt
import csv

# logging.basicConfig(filename='./LOG/main.log')
log = logging.getLogger(__name__)


class Logger(ABC):
    @abstractmethod
    def log_epoch(self, epochs_df):
        raise NotImplementedError("Need to implement the log_epoch function!")


class Printer(Logger):
    """
    Prints output to the terminal using Python's logging module.
    """

    def log_epoch(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        log.info("Epoch {:d}".format(i_epoch))
        last_row = epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            log.info("{:25s} {:.5f}".format(key, val))
        log.info("")


class PrinterTable(Logger):
    """
    Prints output to the terminal using Python's logging module.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.header = dict()
        self.row = dict()
        self.data = None

    def init_table(self, epochs_df):
        last_row = epochs_df.iloc[-1]
        h_id, r_id = 1, 1
        self.header["Set"] = 0
        for key, val in last_row.iteritems():
            labels = key.split('_')
            if labels[0] not in self.row:
                self.row[labels[0]] = r_id
                r_id += 1
            if labels[1] not in self.header:
                self.header[labels[1]] = h_id
                h_id += 1

        self.data = [['--'] * len(self.header) for _ in range(len(self.row) + 1)]
        for k, v in self.header.items():
            self.data[0][v] = k

        for k, v in self.row.items():
            self.data[v][0] = k

    def empty(self):
        assert self.data is not None
        m = len(self.data)
        n = len(self.data[0])
        for i in range(1, m):
            for j in range(1, n):
                self.data[i][j] = '--'
        return

    def log_epoch(self, epochs_df):
        if self.data is None:
            self.init_table(epochs_df)
        else:
            self.empty()

        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        log.info("Epoch {:d}".format(i_epoch))
        last_row = epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            if isinstance(val, float):
                val = round(val, 4)
            labels = key.split('_')
            h_id, r_id = self.header[labels[1]], self.row[labels[0]]
            self.data[r_id][h_id] = val

        tb = pt.PrettyTable()

        tb.field_names = self.data[0]
        for i in range(1, len(self.row) + 1):
            tb.add_row(self.data[i])

        print(tb)
        log.info("")
        with open(self.file_path, 'a+') as log_file:
            log_file.writelines("Epoch {:d}".format(i_epoch) + '\n')
            log_file.write(str(tb) + '\n')
            log_file.writelines('\n')
        return
