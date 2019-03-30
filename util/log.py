#coding=utf-8
'''
Created on 2016年10月12日

@author: dengdan
'''
import datetime
import logging
import util
import sys

def get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S')  

def init_logger(log_file = None, log_path = None, log_level = logging.DEBUG, mode = 'w', stdout = True):
    """
    log_path: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_path is None:
        log_path = '~/temp/log/' 
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.log'
    log_file = util.io.join_path(log_path, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file);
    util.io.make_parent_dir(log_file)
    logging.basicConfig(level = log_level,
                format= fmt,
                filename= util.io.get_absolute_path(log_file),
                filemode=mode)
    
    if stdout:
        console = logging.StreamHandler(stream = sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

#     console = logging.StreamHandler(stream = sys.stderr)
#     console.setLevel(log_level)
#     formatter = logging.Formatter(fmt)
#     console.setFormatter(formatter)
#     logging.getLogger('').addHandler(console)

