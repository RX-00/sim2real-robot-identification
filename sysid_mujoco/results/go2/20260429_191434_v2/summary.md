# MuJoCo SysID v2 Summary

Selected chunk size: `full`
Selected validation RMSE: `0.07148927` rad

## Fit/Validation Split

Training trajectories:
- `/root/sim2real-robot-identification/datasets/go2/traj_0.pt`
- `/root/sim2real-robot-identification/datasets/go2/traj_1.pt`
- `/root/sim2real-robot-identification/datasets/go2/traj_2.pt`
- `/root/sim2real-robot-identification/datasets/go2/traj_3.pt`
- `/root/sim2real-robot-identification/datasets/go2/traj_4.pt`
- `/root/sim2real-robot-identification/datasets/go2/traj_5.pt`
- `/root/sim2real-robot-identification/datasets/go2/traj_6.pt`
- `/root/sim2real-robot-identification/datasets/go2/traj_7.pt`

Validation trajectories:
- `/root/sim2real-robot-identification/datasets/go2/traj_8.pt`
- `/root/sim2real-robot-identification/datasets/go2/traj_9.pt`

## Candidate Validation RMSE

| chunk size | validation RMSE (rad) |
| --- | ---: |
| full | 0.07148927 |
| 100 | 0.07839016 |
| 200 | 0.07359547 |

## Selected Per-Trajectory Validation RMSE

- `traj_8`: `0.07162926` rad
- `traj_9`: `0.07134907` rad

## Selected Per-Joint Validation RMSE

- `FL_hip_joint`: `0.05109534` rad
- `FL_thigh_joint`: `0.06168839` rad
- `FL_calf_joint`: `0.09460802` rad
- `FR_hip_joint`: `0.04921242` rad
- `FR_thigh_joint`: `0.06235407` rad
- `FR_calf_joint`: `0.09414997` rad
- `RL_hip_joint`: `0.05160268` rad
- `RL_thigh_joint`: `0.06106148` rad
- `RL_calf_joint`: `0.09550338` rad
- `RR_hip_joint`: `0.04768586` rad
- `RR_thigh_joint`: `0.06230169` rad
- `RR_calf_joint`: `0.09549837` rad
