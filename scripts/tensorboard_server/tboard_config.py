from tensorboard import program
from multiprocessing import Process
from subprocess import call
from tensorboard import notebook

PORT = None
ENV_CONFIGURED = False

def init_env():
    global ENV_CONFIGURED
    ENV_CONFIGURED = True
    script = \
    """
       pip install -U tensorboard_plugin_profile > /dev/null
       pip install torch_tb_profiler > /dev/null
    """
    call(script, shell=True)

def start_tboard(port=6006):
    global ENV_CONFIGURED
    if not ENV_CONFIGURED: init_env()
    global PORT
    PORT = port
    port = str(port)
    tboard_log = "/content/drive/MyDrive/logs"
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tboard_log, "--port", port])
    url = tb.launch()

def kill_tboard():
  script = \
  f"""
    kill -9 $(lsof -t -i:{PORT})
   """
  call(script, shell=True)

def start_ngrok():
    script = \
    """
      cd /content/drive/MyDrive/DSLearn/scripts/tensorboard_server
      TBOARD_PORT=6006;
      chmod +x ./ngrok;
      ./ngrok authtoken 1rZKWymUNL6m7V4zOsbjoFDJlVW_6o6aaTKYGKNydVxVEFCbA;
      ./ngrok help;
      ./ngrok http $TBOARD_PORT;
    """
    def ngrok_process():
      rc = call(script, shell=True)
    
    NGROK = Process(target=ngrok_process)
    NGROK.start()

def kill_ngrok():
  script = \
  """
    PID=$(ps | grep ngrok | tr -s [:blank:] | cut -d " " -f 2);
    kill -9 $PID;
  """
  call(script, shell=True)