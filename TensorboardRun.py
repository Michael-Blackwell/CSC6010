import tensorflow
import tensorboard
import time

tb = tensorboard.program.TensorBoard()
tb.configure(argv=[None, '--logdir', "Path_To_TensorBoard_File"])
url = tb.launch()

while True:
    time.sleep(1000)
