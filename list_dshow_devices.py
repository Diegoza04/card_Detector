import subprocess

cmd = 'ffmpeg -list_devices true -f dshow -i dummy'
subprocess.call(cmd, shell=True)