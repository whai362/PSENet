import multiprocessing

def cpu_count():
    return multiprocessing.cpu_count()

def get_pool(processes):
    pool = multiprocessing.Pool(processes = processes)
    return pool

def wait_for_pool(pool):
    pool.close()
    pool.join()  

def set_proc_name(name):
    import setproctitle
    setproctitle.setproctitle(name)
    
def kill(pid):
    import util
    if type(pid) == list:
        for p in pid:
            kill(p)
    elif type(pid) == int:
        cmd = 'kill -9 %d'%(pid)
        print cmd
        print util.cmd.cmd(cmd)
    elif type(pid) == str:
        pids = get_pid(pid)
        kill(pids)
    else:
        raise ValueError, 'Not supported parameter type:', type(pid)

def ps_aux_grep(pattern):
    import util
    cmd = 'ps aux|grep %s'%(pattern)
    return util.cmd.cmd(cmd)

        
def get_pid(pattern):
    import util
    cmd = 'ps aux|grep %s'%(pattern)
    results = util.cmd.cmd(cmd)
    results = util.str.split(results, '\n')
    pids = []
    for result in results:
        info = result.split()
        if len(info) > 0:
            pid = int(info[1])
            pids.append(pid)
    return pids

