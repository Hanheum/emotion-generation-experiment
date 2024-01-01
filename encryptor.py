#0: 듀   1: 듇

txt = input('txt:')
txt = txt.encode('utf-8')

bits = ''.join(format(byte, '08b') for byte in txt)

returning_txt = ''
for bit in bits:
    if bit == '0':
        returning_txt += '듀'
    else:
        returning_txt += '듇'

print(returning_txt)