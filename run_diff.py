#!/usr/bin/env python3
"""Run the l3 components diff beverin_l3_dequant2 vs clariden ref, output to a file."""
import subprocess, sys

CMD = ("python3 /capstor/scratch/cscs/xyao/opentela-cookbook/meta/diag/glm53/comp_diff.py "
       "components /capstor/scratch/cscs/xyao/glm-53-flash-beverin/comp_capture/beverin_l3_dequant2 "
       "--clariden /capstor/scratch/cscs/xyao/glm-53-flash-beverin/comp_capture/clariden_l3_dequant_ref "
       "--layer 3")

out_path = "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/logs/comp_diff_625913.out"
with open(out_path, "w") as fh:
    rc = subprocess.call(CMD, shell=True, stdout=fh, stderr=subprocess.STDOUT)
print("rc:", rc, "->", out_path)
