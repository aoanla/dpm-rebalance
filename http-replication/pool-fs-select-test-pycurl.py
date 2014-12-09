#!/usr/bin/python

import argparse
import pycurl
import sys
import urllib


parser = argparse.ArgumentParser()
parser.add_argument('filename', help="/dpm/cern.ch/home/dteam/puttest.X")
parser.add_argument('--querystring', dest='querystring',
    help="?pool=pool01&filesystem=dpmdisk01.cern.ch:/srv/dpm/01&replicate",
    default='')
parser.add_argument('--head', dest='headnode',
    help="dpmhead01.cern.ch", default="dpmhead01.cern.ch")
parser.add_argument('-E', dest='cert',
    help="The x509 client certificate")
parser.add_argument('-t', dest='testing', action='store_true',
    help="Don't execute the copy, just print the destination address",
    default=False)

args = parser.parse_args()

class Response(object):
  """ utility class to collect the response """
  def __init__(self):
    self.chunks = []
  def callback(self, chunk):
    self.chunks.append(chunk)
  def content(self):
    return ''.join(self.chunks)
  def headers(self):
    s = ''.join(self.chunks)
    print s
    header_dict = {}
    for line in s.split('\r\n'):
      try:
        key,val = line.split(':',1)
        header_dict[key] = val
      except:
        pass

    return header_dict


res = Response()

headnode = args.headnode
certificate = args.cert

c = pycurl.Curl()
c.setopt(c.SSLCERT, certificate)
c.setopt(c.HEADERFUNCTION, res.callback)
c.setopt(c.SSL_VERIFYPEER, 0)
c.setopt(c.FOLLOWLOCATION, 0)
c.setopt(c.SSL_VERIFYHOST, 0)
c.setopt(c.CUSTOMREQUEST, 'PUT')
c.setopt(c.URL, 'https://'+headnode+args.filename+args.querystring)
c.perform()

print res.headers()

dest_redir_location = res.headers()['Location']
dest_redir_location = urllib.unquote(dest_redir_location)
print 'DESTINATION phys location', dest_redir_location
print
if args.testing:
  sys.exit(0)

res2 = Response()
c.setopt(c.SSLCERT, certificate)
c.setopt(c.HEADERFUNCTION, res2.callback)
c.setopt(c.SSL_VERIFYPEER, 0)
c.setopt(c.SSL_VERIFYHOST, 0)
c.setopt(c.CUSTOMREQUEST, 'COPY')
c.setopt(c.HTTPHEADER, ['Destination: '+dest_redir_location,
                        'X-No-Delegate: true'])
c.setopt(c.FOLLOWLOCATION, 1)
c.setopt(c.URL, 'https://'+headnode+args.filename)
c.perform()
