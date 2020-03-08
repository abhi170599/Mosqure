import os 
import signal 
import subprocess 


BASE_PATH = '../'

SERVER_DIR = BASE_PATH+'server'

CLASS_MODEL_DIR = BASE_PATH+'image_classifier'



os.chdir(SERVER_DIR)
tf_ic_server = ""
flask_server = ""

try:
    tf_ic_server = subprocess.Popen(["tensorflow_model_server ",
                                     "--model_base_path={}".format(CLASS_MODEL_DIR),
                                     "--rest_api_port=8000 --model_name=ImageClassifier"],
                                    stdout=subprocess.DEVNULL,
                                    shell=True,
                                    preexec_fn=os.setsid)
    print("Started TensorFlow Serving ImageClassifier server!")

    flask_server = subprocess.Popen(["export FLASK_ENV=development && flask run --host=0.0.0.0"],
                                    stdout=subprocess.DEVNULL,
                                    shell=True,
                                    preexec_fn=os.setsid)
    print("Started Flask server!")

    while True:
        print("Type 'exit' and press 'enter' OR press CTRL+C to quit: ")
        in_str = input().strip().lower()
        if in_str == 'q' or in_str == 'exit':
            print('Shutting down all servers...')
            os.killpg(os.getpgid(tf_ic_server.pid), signal.SIGTERM)
            os.killpg(os.getpgid(flask_server.pid), signal.SIGTERM)
            print('Servers successfully shutdown!')
            break
        else:
            continue
except KeyboardInterrupt:
    print('Shutting down all servers...')
    os.killpg(os.getpgid(tf_ic_server.pid), signal.SIGTERM)
    os.killpg(os.getpgid(flask_server.pid), signal.SIGTERM)
    print('Servers successfully shutdown!')