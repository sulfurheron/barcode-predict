import os

if not os.path.isdir("saved_models"):
    try:
        os.mkdir("saved_models")
    except OSError as e:
        print("Error creating `saved_models` directory: {}".format(e.strerror))

if not os.path.isdir("logs"):
    try:
        os.mkdir("logs")
    except OSError as e:
        print("Error creating `logs` directory: {}".format(e.strerror))
