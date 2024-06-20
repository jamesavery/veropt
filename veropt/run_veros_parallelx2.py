import os, sys, subprocess, json
import numpy as np

from multiprocessing import Process

# venv_path, run_path, device_id, setup_script, setup_args = sys.argv[1:]

def activate_venv(venv_path):
    source = f'source {venv_path}/bin/activate'
    dump = 'python3 -c "import os, json;print(json.dumps(dict(os.environ)))"'
    pipe = subprocess.Popen(['/bin/bash', '-c', '%s && %s' %(source,dump)], stdout=subprocess.PIPE)
    env = json.loads(pipe.stdout.read())
    os.environ = env
    

def run_veros(run_path, setup_script, setup_args, venv_path, device_id=1, float_type="float32", use_gpu=True):
    activate_venv(venv_path)
    os.chdir(run_path)
    HOME = os.environ['HOME']    

    print(f"We are in {os.getcwd()}, veros={HOME}/.local/bin/veros")
    env_copy = os.environ.copy()
    env_copy["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # print(device_id)
    # print(env_copy)

    gpu_string = "--backend jax --device gpu" if use_gpu else ""
        
    command = f"{HOME}/.local/bin/veros run {gpu_string} --float-type {float_type} --setup-args {setup_args} {setup_script}".split(' ')
    print(command)
    pipe    = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env_copy)
    output, errors = pipe.communicate()
    print(f"output: {output}")
    print(f"errors: {errors}")


def run_two(run_path, setup_script, setup_args0, setup_args1, venv_path, float_type="float32", use_gpu=True):
    
    args0 = (run_path, setup_script, setup_args0, venv_path, 0, float_type, use_gpu)
    args1 = (run_path, setup_script, setup_args1, venv_path, 1, float_type, use_gpu)
    print(setup_args0)
    print(setup_args1)

    p0 = Process(target=run_veros, args=args0)
    p0.start()
    p1 = Process(target=run_veros, args=args1)
    p1.start()
    p0.join()
    p1.join()
