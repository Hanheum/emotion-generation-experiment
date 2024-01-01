decrypted_txt = input('decrypted txt:')

binary = ''
for i in decrypted_txt:
    if i == '듀':
        binary += '0'
    else:
        binary += '1'

txt = int(binary, 2).to_bytes((len(binary)+7)//8, 'big')
txt = txt.decode()
print(txt)