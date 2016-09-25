
#
# wrap comment lines
#
# issues:
# - does not detect comments running beyond 1 line
# - does not detect indented comments
#

fr=open('s4b01.py', 'r')
fw=open('s4b01-wrap.py', 'w')
data = fr.readlines()
for lnr in data:
  if lnr[0:1]!='#':
    fw.write(lnr)
    continue
  words = lnr.split()
  lnw=words[0]
  for w in words[1:]:
    if len(lnw)+len(w)<69:
      lnw=lnw+' '+w
    else:
      fw.write(lnw+'\n')
      lnw='# '+w
  if lnw!='':
      fw.write(lnw+'\n')
fr.close()
fw.close()
# eop

   