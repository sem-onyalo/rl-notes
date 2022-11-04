from gym.envs.registration import register
from gym.envs.registration import registry

gyms = [
    { 
        "id": "sem_gyms/DriftCarEnv-v0", 
        "entry_point": "mdp.sem_gyms.drift_car_env:DriftCarEnvV0", 
        # "max_episode_step": 1000
    }
]

for gym in gyms:
    if not gym["id"] in registry:
        register(
            id=gym["id"],
            entry_point=gym["entry_point"],
            # max_episode_steps=gym["max_episode_step"]
        )
