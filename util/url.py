import sys
import os
from six.moves import  urllib

import util
def download(url, path):
    filename = path.split('/')[-1]
    if not util.io.exists(path):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r-----Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      path, _ = urllib.request.urlretrieve(url, path, _progress)
      print()
      statinfo = os.stat(path)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    