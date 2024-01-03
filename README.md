My personal experiments to make emotion using Deep Reinforcement Learning.

Gridworld: practice env

emotion_v0: test env

            -red polygons: enemies

            -white polygon: agent

            -green dots: foods(restore health)
            
            -blue dot: helper(neutralize enemy)

emotion_vo_DQN.py: test env implementation

    While I wrote DQN on the title of the code, it's not really traditional DQN.
    I expect agent to be able to do sophisticated tasks by implementing "memory vector", rather than just react to the state.
    "Memory vector" represents the synapse between neurons, and it is remembered. 
    Thus, agent can know required informations to complete the task, without actually storing every single steps(like Transformers), or stacking states(DQN with image inputs).
    The DQN is currently used for this project, but it can be changed into other RL methods, such as Policy Gradient, depending on situations.

When agent touches food, it eats the food and gets instant reward. But it doesn't get reward when it touches helper. 
But if agent hits enemy while carrying helper, it gets more reward than food. 
Goal of this project is to approximate agent's behaviour to obtain helper, even it doesn't give reward.
I personally consider this can be called as primative level of attachment. 

contact: chh3653@gmail.com
