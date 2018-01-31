# Simple Road Scenario

## Installation

```
git clone https://github.com/cyrilibrahim/simple_road_ppo/
chmod u+x simple_road_ppo/circuit_linux/circuit_linux.x86_64
```

## Run on Slurm

```
sinter -c 2 --mem=16000 --gres=gpu --reservation=res_stretch

LD_LIBRARY_PATH=/Tmp/lisa/os_v5/lib/glx:$LD_LIBRARY_PATH xvfb-run -n $SLURM_JOB_ID -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" ./simple_road_ppo/circuit_linux/circuit_linux.x86_64
```
