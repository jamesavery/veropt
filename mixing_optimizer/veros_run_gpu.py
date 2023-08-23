import os, sys, subprocess, json

#venv_path, run_path, device_id, setup_script, setup_args = sys.argv[1:]

def activate_venv(venv_path):
    source = f'source {venv_path}/bin/activate'
    dump = 'python3 -c "import os, json;print(json.dumps(dict(os.environ)))"'
    pipe = subprocess.Popen(['/bin/bash', '-c', '%s && %s' %(source,dump)], stdout=subprocess.PIPE)
    env = json.loads(pipe.stdout.read())
    os.environ = env
    

def run_veros_gpu(venv_path, run_path, device_id, setup_script, setup_args, float_type="float32"):
    activate_venv(venv_path)
    os.chdir(run_path)
    print(f"We are in {os.getcwd()}")
    command = f"/home/avery/.local/bin/veros run --backend jax --device gpu --float-type {float_type} --setup-args {setup_args} {setup_script}".split(' ')
    pipe    = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = pipe.communicate()
    print(f"output: {output}")
    print(f"errors: {errors}")
