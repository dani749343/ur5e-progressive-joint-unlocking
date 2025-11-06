from setuptools import setup, find_packages

setup(
    name='homestri_ur5e_rl',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'gymnasium',
        'gymnasium-robotics',
        'mujoco',
        'stable_baselines3',
        "stable-baselines3[extra]",
        'syllabus_rl',
        'pynput',
        'torch',
    ],
    # Other information like author, author_email, description, etc.
)
