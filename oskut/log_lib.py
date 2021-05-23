import logging
from datetime import date,time
def set_log(info1='',info2=date.today()):
    logging.basicConfig(filename=f'{info1}.{info2}.log',level=logging.DEBUG)

def save_log(text): 
    logging.debug(text)

def print_log(text):
    logging.info(text)

def log(text):
    print_log(text)
    save_log(text)
