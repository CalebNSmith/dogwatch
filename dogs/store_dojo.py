import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # !{ERROR,WARNING,INFO}
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from db.hdf5_stuff import dojo_store, dojo_read

#dojo_store()
dojo_read()
