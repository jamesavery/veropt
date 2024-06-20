import os, sys, subprocess, json

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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    gpu_string = "--backend jax --device gpu" if use_gpu else ""
        
    command = f"/home/marta/.local/bin/veros run {gpu_string} --float-type {float_type} --setup-args {setup_args} {setup_script}".split(' ')
    pipe    = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = pipe.communicate()
    print(f"output: {output}")
    print(f"errors: {errors}")
