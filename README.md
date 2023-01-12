# Stego
please check it in raw mode


This project is based on the "https://github.com/aamixsh/VStegNET" 
we have made some changes.
  1.correct some errors of code
  2.add a test module for revealing_net
  3.add a statement about requirement and setting dataset
if you want to test or train the project
  1.install the environment as follows
    tensorflow-gpu = 1.10.0
    numpy = 1.14.5
    matplotlib = 2.1.0
    scipy= 1.0.0
    keras = 2.2.0
    opencv=3.4
    sklearn=0.20.0
  2.set the test and train dataset at right place by right way
    for example
    1.change the settings of directory in stego_net.py
      test_container_loc = 'ucf/test_rel/'#the directory of data for test the revealing_net
      test_loc = 'ucf/test/'#the directory of data for test the stego_net
      train_loc = 'ucf/train/'#the directory of data for train the stego_net
    2.put the dataset like:
      ucf
        -test_rel
          -video1
            -container
              -0.jpg
              -1.jpg
              -2.jpg
              -3.jpg
              -4.jpg
              -5.jpg
              -6.jpg
              -7.jpg
              ...
            -cover
              -0.jpg
              -1.jpg
              -2.jpg
              -3.jpg
              -4.jpg
              -5.jpg
              -6.jpg
              -7.jpg
              ...
            -secret
              -0.jpg
              -1.jpg
              -2.jpg
              -3.jpg
              -4.jpg
              -5.jpg
              -6.jpg
              -7.jpg
              ...
          -video2
          ....
        -test
          -video1
            -0.jpg
            -1.jpg
            -2.jpg
            -3.jpg
            -4.jpg
            -5.jpg
            -6.jpg
            -7.jpg
            ...
          -video2
          -....
        -train
          -video1
            -0.jpg
            -1.jpg
            -2.jpg
            -3.jpg
            -4.jpg
            -5.jpg
            -6.jpg
            -7.jpg
            ...
          -video2
          -....
      attention:please exactra the frames from the video and rename them
    3.execute the code in the end of "stego_net.py"
      there are three statement for test
      1.if you want to train the module:
        please only execute the code "m.train()"
      2.if you want to test the module:
        please only execute the code "m.test()"
      3.if you want to test the revealing_net of the module:
        please only execute the code "m.test_reveal()"
 we have a simple example for learn how to set the dataset and include a checkpoint file,
 you can get it on:"link：https://pan.baidu.com/s/1G9yUbsuzeicFknzBqAXvNg?pwd=8888 
 pwd=8888 

