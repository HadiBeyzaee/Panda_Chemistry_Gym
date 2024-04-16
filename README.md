<!-- Title and Introduction -->
# Insertion-Scenarios
**Exploring Deep Reinforcement Learning for Precise Cap, Vial, and Rack Insertions using Stable Baselines3 Algorithms**

Welcome to the Insertion-Scenarios project, where we dive deep into the world of reinforcement learning and precision insertions. In this repository, we showcase cutting-edge scenarios involving the delicate tasks of Cap, Vial, and Rack insertions, all accomplished using advanced techniques from the stable-baselines3 and sb3-contrib library.

## Project Overview
As industries demand higher precision and efficiency in their automation processes, the need for intelligent and adaptive robotic systems becomes ever more critical. This project serves as a playground for developing and testing reinforcement learning algorithms that can excel at intricate insertion tasks.

Our primary focus lies in three fundamental insertion scenarios:

1. **Vial Capping Pick and Place:** Teaching robots to accurately place caps on vials.
2. **Vial Insertion:** Navigating the intricate process of inserting vials into different holders and racks.
3. **Rack Insertion:** Mastering the skill of inserting racks, both empty and loaded, into their designated holders.

By leveraging the power of stable-baselines3 and sb3-contrib, we've created a platform for experimenting with different reinforcement learning algorithms and methodologies, fine-tuning their parameters, and ultimately training models that exhibit remarkable precision in these challenging tasks.

---

<!-- Placing Cap on Vial -->
<div align="center">
  <h2><strong>Vial Capping Task</strong></h2>
</div>


<div align="center">
  <table>
    <tr>
      <th>Placing Cap on Vial</th>
    </tr>
    <tr>
      <td align="center">
        <img src="https://github.com/HadiBeyzaee/Chemistry_lab_tool_insertions_DRL/assets/124352983/65449a0e-8364-4773-8c30-8415a0969772" alt="Placing Cap on Vial" width="500px">
      </td>
    </tr>
  </table>
</div>



<br> <!-- Add spacing between scenarios -->
<!-- Vial Insertion Tasks -->
<div align="center">
  <h2><strong>Vial Insertion Tasks</strong></h2>
</div>

<div align="center">
  <table>
    <tr>
      <th>From Single Holder into Single Holder</th>
      <th>From Single Holder into Rack</th>
    </tr>
    <tr>
      <td align="center">
        <img src="https://github.com/HadiBeyzaee/Chemistry_lab_tool_insertions_DRL/assets/124352983/5aebe871-2e3d-477a-9a34-e0e9426e265e" alt="Vial Insertion: Single Holder into Single Holder" width="400px">
      </td>
      <td align="center">
        <img src="https://github.com/HadiBeyzaee/Chemistry_lab_tool_insertions_DRL/assets/124352983/c963c138-4e01-478f-9c27-b9fbc53b310a" alt="Vial Insertion: Single Holder into Rack" width="400px">
      </td>
    </tr>
    <tr>
      <th>From Single Holder into Loaded Rack</th>
      <th>From Rack into Rack</th>
    </tr>
    <tr>
      <td align="center">
        <img src="https://github.com/HadiBeyzaee/Chemistry_lab_tool_insertions_DRL/assets/124352983/ea943317-4e41-4151-b3f9-9661ec0892c1" alt="Vial Insertion: Holder into Loaded Rack" width="400px">
      </td>
      <td align="center">
        <img src="https://github.com/HadiBeyzaee/Chemistry_lab_tool_insertions_DRL/assets/124352983/80952208-c1b4-4cdb-b0ad-447e7709513a" alt="Vial Insertion: Rack into Rack" width="400px">
      </td>
    </tr>
    <tr>
      <th>From Loaded Rack into Rack</th>
      <th>From Rack into Loaded Rack</th>
    </tr>
    <tr>
      <td align="center">
        <img src="https://github.com/HadiBeyzaee/Chemistry_lab_tool_insertions_DRL/assets/124352983/ca5df7fd-68c0-49d9-b075-8394206be4e9" alt="Vial Insertion: Loaded Rack into Rack" width="400px">
      </td>
      <td align="center">
        <img src="https://github.com/HadiBeyzaee/Chemistry_lab_tool_insertions_DRL/assets/124352983/eebddfb2-e2c1-4456-bb65-7b6c4c465799" alt="Vial Insertion: Rack into Loaded Rack" width="400px">
      </td>
    </tr>
  </table>
</div>

<br> <!-- Add spacing between scenarios -->

<!-- Rack Insertion Tasks -->
<div align="center">
  <h2><strong>Rack Insertion Tasks</strong></h2>
</div>

<div align="center">
  <table>
    <tr>
      <th>Empty Rack</th>
      <th>Loaded Rack</th>
    </tr>
    <tr>
      <td align="center">
        <img src="https://github.com/HadiBeyzaee/Chemistry_lab_tool_insertions_DRL/assets/124352983/ec33343e-858c-4063-87b8-1f79da04e425" alt="Rack Insertion: Empty Rack" width="400px">
      </td>
      <td align="center">
        <img src="https://github.com/HadiBeyzaee/Chemistry_lab_tool_insertions_DRL/assets/124352983/bdc02d08-d5d6-40a6-886d-2381efaf8cf3" alt="Rack Insertion: Loaded Rack" width="400px">
      </td>
    </tr>
  </table>
</div>


<!-- Installation Requirements -->
## Installation Requirements

Before starting the Insertion Scenarios project, ensure that you have the necessary packages installed. You can use the following commands to set up your environment:

### Create a new Conda environment
```bash
conda create -n your_environment_name python=3.8
```

### Activate the Conda environment
```bash
conda activate your_environment_name
```

### Install stable-baselines3
```bash
pip install stable-baselines3==1.6.0
```

### Install sb3-contrib
```bash
pip install sb3-contrib==1.6.0
```

### Install TensorBoard
```
pip install tensorboard
```

### Install gym-robotics
```bash
pip install gym-robotics==0.1.0
```

### Install gym
```bash
pip install gym==0.21.0
# Alternatively if typo opencv-python>=3 happened:
# pip install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40
```


## Troubleshooting
If you run into any errors during the training, please ensure that you are using the exact version of the packages specified. You can check the installed package versions using the following command:

```bash
pip show package_name
```
For any error regarding gym version see : https://github.com/openai/gym/issues/3202

