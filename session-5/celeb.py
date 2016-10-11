
import numpy as np
import os
import os
import urllib.request
#os.mkdir('img_align_celeba_2')

# Now perform the following 10 times:
for img_i in range(156, 2000):

    # create a string using the current loop counter
    f = '000%03d.jpg' % img_i

    # and get the url with that string appended the end
    url = 'https://s3.amazonaws.com/cadl/celeb-align/' + f

    # We'll print this out to the console so we can see how far we've gone
    print(url, end='\r')

    # And now download the url to a location inside our new directory
    try:
      urllib.request.urlretrieve(url, os.path.join('img_align_celeba_2', f))
    except Exception as e:
        print(f, " EXC: ",e)


# Using the `os` package, we can list an entire directory.  The documentation or docstring, says that `listdir` takes one parameter, `path`:

# In[10]:
