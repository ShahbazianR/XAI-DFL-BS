# 📌 Deep Reinforcement Learning for Base Station Management

This repository implements a **Deep Reinforcement Learning (DRL) framework** for optimizing the energy efficiency of base stations in mobile networks. It utilizes **Deep Q-Networks (DQN), Prioritized DQN (PDQN), and traditional Q-learning** for decision-making on active/sleep modes of base stations. Additionally, **ARIMA** models are used for time-series traffic forecasting.

## 📖 **Citation**
If you use this framework in your research, please cite:

N. Movahedkor and R. Shahbazian, "Decentralized Federated Deep Reinforcement Learning Framework for Energy-Efficient Base Station Switching Control,"  
2024 11th International Symposium on Telecommunications (IST), Tehran, Iran, Islamic Republic of, 2024, pp. 455-460,  
doi: [10.1109/IST64061.2024.10843518](https://doi.org/10.1109/IST64061.2024.10843518).

**Keywords:** Training, Base stations, Energy consumption, Privacy, 5G mobile communication, Federated learning, Switches, Quality of service, Deep reinforcement learning, Telecommunications, Base Station Switching Management, Traffic Forecasting, Decentralized Federated Learning, Deep Reinforcement Learning, Energy Efficient Communication Networks.
---

## 📁 **Project Structure**
```
├── DFRL_Main.py          # Main script for running the DRL framework
├── Algorithm.py          # Implementation of the FRL (Federated Reinforcement Learning) function
├── Agent.py              # Base station agent managing energy efficiency
├── PDQN_Network.py       # Prioritized Deep Q-Network implementation
├── TP_Network.py         # Traffic prediction model using BiLSTM and Attention
├── TP_Baseline_models.py # Alternative CNN-LSTM traffic prediction models
├── Q_Baseline.py         # Q-learning baseline implementation
├── DQN_SW_Baseline.py    # Standard DQN baseline with shared weights
├── ARIMA.py              # ARIMA-based traffic prediction
├── ARIMA_Scaled.py       # Scaled ARIMA evaluation
└── utils.py              # Utility functions for data loading and preprocessing
```

---

## 🛠 **Installation & Setup**
### **1️⃣ Prerequisites**
- Python (>= 3.7)
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- tqdm
- Statsmodels

### **2️⃣ Installation**
Clone the repository and install dependencies:

```bash
git clone [https://github.com/ShahbazianR/XAI-DFL-BS/]
cd *repo
pip install -r requirements.txt
```

---

## 🚀 **Usage**
### **1️⃣ Train the Deep Reinforcement Learning Model**
Run the main script to start training:

```bash
python DFRL_Main.py
```

### **2️⃣ Traffic Prediction with ARIMA**
To run ARIMA-based forecasting:

```bash
python ARIMA.py
```

### **3️⃣ Run Different RL Algorithms**
- **Prioritized Deep Q-Network (PDQN):**
  ```bash
  python PDQN_Network.py
  ```
- **Standard Deep Q-Network (DQN):**
  ```bash
  python DQN_SW_Baseline.py
  ```
- **Q-Learning Baseline:**
  ```bash
  python Q_Baseline.py
  ```

---

## 📊 **Model Description**
### **1️⃣ Traffic Prediction Model**
- Implements **BiLSTM-Attention and CNN-LSTM** for predicting future base station traffic.
- **ARIMA-based models** are also available for time-series forecasting.

### **2️⃣ Reinforcement Learning Framework**
- Uses **Deep Q-Learning (DQN) and Prioritized DQN (PDQN)** to optimize base station activation.
- Considers **energy cost, QoS degradation, and switching cost** in the reward function.

---

## 📌 **Results & Logs**
Training and evaluation results are saved in automatically generated folders like:

```
DFRL_10_20250311-153000/
├── Experiment_Config.txt        # Configurations of the experiment
├── Average_TimeStep_Results.csv # Average energy consumption, QoS, and switching cost
└── Training_Ep_10.csv           # Training results per episode
```

---

## 🤝 **Contributing**
Contributions are welcome! Please:
1. Fork this repository
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

---

## 📝 **License**
This project is open-source and available under the **MIT License**.

---


🔥 **Star this repo if you found it useful! 🚀**
