import pylikwid

pinfo = pylikwid.getpowerinfo()
PKG_DOMAIN_ID = pinfo['domains']['PKG']['ID']
CORES = [0, 1, 2, 3, 4, 5, 6]


def start_power():
    ret = []
    for core in CORES:
        ret.append(pylikwid.startpower(core, PKG_DOMAIN_ID))
    return ret


def stop_power():
    ret = []
    for core in CORES:
        ret.append(pylikwid.stoppower(core, PKG_DOMAIN_ID))
    return ret


def get_power_sum(start, stop):
    assert len(start) == len(stop)

    pwr_sum = 0
    for i in range(0, len(start)):
        pwr_sum += pylikwid.getpower(start[i], stop[i], PKG_DOMAIN_ID)

    return pwr_sum
