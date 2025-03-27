def int_as_f8(a):
    offset = 8
    m = a & 0x7
    e = (a >> 3) & 0xf
    s = (a >> 7) & 1
    if e == 0:
        v = m / 8 * (2**(1 - offset))
    else:
        v = (1 + m / 8) * (2**(e - offset))
    if s:
        v *= -1
    return v
