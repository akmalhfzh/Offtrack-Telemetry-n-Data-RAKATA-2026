## 1. System definition (what MPC controls)

### State vector

Minimal and sufficient:  
[  
x = [v,\ s,\ SOC]  
]

- (v): vehicle speed
    
- (s): distance along track
    
- (SOC): state of charge (or energy proxy)
    

### Control

[  
u \in {0,\ u_{max}}  
]  
or continuous torque if available.

Binary throttle is common in SEM. Continuous is fine if actuators allow it.

---

## 2. Dynamics (digital twin)

Discrete-time (or distance-based):

[  
\begin{aligned}  
v_{k+1} &= v_k + \Delta t \cdot a(v_k, u_k) \  
s_{k+1} &= s_k + \Delta t \cdot v_k \  
SOC_{k+1} &= SOC_k - \frac{(P_{phys} + \hat{\Delta P}_{NARX})\Delta t}{E_{bat}}  
\end{aligned}  
]

This explicitly shows **where NARX enters**: energy only.

---

## 3. MPC cost function (this defines “strategy”)

### Stage cost

[  
J_k =  
w_E \cdot (P_{phys}(x_k,u_k) + \hat{\Delta P}_{NARX})

- w_v \cdot (v_k - v_{ref})^2
    
- w_u \cdot u_k^2  
    ]
    

Interpretation:

- (w_E): energy minimization → glide behavior
    
- (w_v): speed band enforcement
    
- (w_u): smoothness / actuator protection
    

---

### Terminal cost (critical)

[  
J_N = w_T \cdot \max(0, T_{pred} - T_{max})^2

- w_{SOC} \cdot (SOC_N - SOC_{target})^2  
    ]
    

This enforces:

- lap-time constraint
    
- finish SOC buffer
    

Without terminal terms, MPC cheats.

---

## 4. Constraints (non-negotiable)

[  
\begin{aligned}  
v_{min} \le v_k \le v_{max} \  
u_k \in {0,u_{max}} \  
T_{pred} \le T_{max} + \epsilon  
\end{aligned}  
]

These constraints, not rewards, enforce race rules.

---

## 5. How strategy emerges

- High (w_E) → long glide
    
- High (w_v) → narrow speed band
    
- Tight (T_{max}) → aggressive accel bursts
    

No heuristic rules.  
The glide–accelerate pattern is an **optimal solution**, not programmed behavior.

---

## 6. Which MPC parameters are tunable (and which are fixed)

### Fixed (do NOT learn)

- dynamics
    
- constraints
    
- horizon discretization
    
- actuator limits
    

### Learnable (small set)

[  
\theta = [w_E,\ w_v,\ w_u,\ w_{SOC},\ w_T]  
]

5 parameters max.  
Anything more is unjustified.

---

## 7. RL for tuning MPC weights (outer loop)

### Abstraction

- Episode = one lap
    
- Action = choose (\theta)
    
- Environment = digital twin + MPC
    
- Reward = lap-level metric
    

---

### Reward (correct form)

[  
R = -E_{lap}  
\quad \text{s.t. } T_{lap} \le T_{max}  
]

If constraint violated:  
[  
R = -E_{lap} - M  
]  
with large fixed (M).

No shaping. No ratios. No hacks.

---

## 8. Which “RL” algorithm to use (be precise)

Use **black-box optimization**, not policy networks:

- CMA-ES (best choice)
    
- Bayesian optimization
    
- Cross-entropy method
    

This is episodic parameter search, not control RL.

Call it:

> “learning-based MPC weight tuning”

Not “Deep RL controller”.

---

## 9. Evaluation protocol (this is what reviewers look for)

Compare four cases:

1. Physics MPC only
    
2. Physics + NARX residual MPC
    
3. Tuned MPC (no ML residual)
    
4. Tuned MPC + NARX residual (best)
    

Metrics:

- km/kWh
    
- lap time
    
- constraint violations
    
- robustness to slope / wind variation
    

---

## 10. How to write this safely in the paper

Correct phrasing:

> “A supervised NARX model is employed to learn residual power losses from telemetry, enhancing the fidelity of a physics-based digital twin. An MPC controller performs energy-optimal driving under strict time constraints. An outer-loop learning agent tunes MPC cost weights at the lap level.”

This passes review.

---

## 11. Absolute red lines (do not cross)

- RL selecting throttle
    
- NARX trained on km/kWh
    
- Reward-shaped instantaneous efficiency
    
- Online adaptive learning without stability bounds
    

Any of these = rejection.

---

## Final mental model (lock this in)

- **NARX** → corrects physics
    
- **MPC** → decides actions (strategy)
    
- **RL** → tunes MPC priorities
    
- **Physics** → always in charge
    