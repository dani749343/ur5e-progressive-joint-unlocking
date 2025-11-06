# Progressive Joint Unlocking
## Overview
This algorithm progressively introduces joints based on curricula in the UR5e environment. This environment is built on Ian Chuang's homestri-ur5e-rl https://github.com/ian-chuang/homestri-ur5e-rl. This code performs a reaching task decomposed into
an **arc** phase and a **descent** phase. For both phases of the task the ascending progressive joint unlocking order and curricula for joint progression can be adjusted for both phases.

## Training
To begin training, open the pju_final_train.py file in the scripts/PJU_Final directory. The joint unlock progression based on DoF introduced can be adjust for the arc with ARC_DOF_SEQUENCE, and for the descent with DESCENT_DOF_SEQUENCE. The respective
tolerances required for progression can also be adjusted for the arc with ARC_STAGE_CONFIGS and for the descent with DESCENT_STAGE_CONFIGS. Trainig can be continued from a previous model by setting LOAD_PREVIOUS_MODEL True and selecting the desired 
phase with STARTING_PHASE and stage with STARTING_STAGE.

## Testing
To test a model that has been trained, open the pju_final_test.py file in the scripts/PJU_Final directory. Adjust the testing phase with TEST_PHASE and testing stage with TEST_STAGE. Phase options for testing are **arc**, **descent**, and if full 
training has been completed **cascade** to simulate the entire arc and descent movement in one window.
