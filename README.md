A machine learning approach to determine if your dog is standing, sitting, or laying.

The dogs directory contains all of the python code that handles training and prediction (using TensorFlow). It also contains a websocket that will offer predictions from a RTSP/UDP stream and forward the results to another endpoint (e.g. android device).

The android directory contains the android application that offers 
1. Local prediction from a converted TensorFlowLite model
2. Streaming the camera to a server for prediction
3. Connecting to a webcam and receiving the predictions simultaneously
