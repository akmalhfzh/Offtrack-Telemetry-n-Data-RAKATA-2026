## 1. Does MPC output the driving strategy?

**Yes.**  
But **strategy ≠ policy ≠ heuristic**.

### What MPC actually outputs

At every control step, MPC solves:

[  
\min_{u_{0..N-1}} \sum_{k=0}^{N-1} J(x_k, u_k)  
]

subject to:

- vehicle dynamics
    
- speed limits
    
- time / lap constraints
    
- actuator limits
    

The **output** is:

- the _first_ control action (u_0)
    
- e.g. throttle ON / OFF, torque value
    

Because this is repeated every step, the **emergent behavior** is:

- accelerate
    
- cut throttle
    
- glide
    
- re-accelerate
    

So:

- MPC does **not** explicitly encode “glide strategy”
    
- glide emerges because energy-optimal solutions force it
    

That’s why MPC beats heuristics.

---

## 2. Where MPC sits conceptually

Think in layers:

- **Physics + constraints** define what’s possible
    
- **Cost function** defines what’s desirable
    
- **MPC** computes the action sequence
    

If you change weights, you change strategy.

---

## 3. What “strategy” really means here

In SEM-style driving, strategy is:

- speed band
    
- acceleration aggressiveness
    
- glide duration
    
- buffer to lap-time constraint
    

MPC encodes all of these **implicitly** via:

- cost weights
    
- constraints
    
- horizon length
    

No rules needed.

---

## 4. Using RL to tune MPC weights (this is valid)

This is **not** Deep RL control.  
This is **meta-optimization**.

### Correct abstraction

- Inner loop: MPC (fast, deterministic)
    
- Outer loop: RL (slow, episodic)
    

```
RL → chooses MPC weights
MPC → controls vehicle
Environment → returns lap energy + time
```

RL never touches actuators.

---

## 5. What RL actually tunes (concrete)

Typical tunable parameters:

- energy vs time weight
    
- terminal SOC penalty
    
- speed deviation penalty
    
- horizon length (sometimes)
    

RL action space is **tiny** (3–6 parameters).

---

## 6. Why RL works here (but not for control)

Because:

- low-dimensional action space
    
- sparse rewards (lap-level)
    
- no safety-critical exploration online
    
- MPC enforces feasibility
    

RL can be slow and sloppy — MPC contains the damage.

---

## 7. What RL algorithm is appropriate

Do **not** use heavy Deep RL.

Use:

- Bayesian optimization
    
- CMA-ES
    
- episodic policy gradient
    
- bandit-style RL
    

Calling this “RL” is fine, but it’s closer to black-box optimization.

---

## 8. What NOT to do

- RL selecting throttle directly
    
- RL inside the MPC loop
    
- RL tuning constraints online
    

All unsafe.

---

## 9. How this fits your paper

You can state:

> “An outer-loop learning agent tunes the MPC cost weights to adapt energy–time trade-offs across tracks, while the inner-loop MPC guarantees constraint satisfaction.”

Reviewers accept this.

---

## Bottom line

- MPC outputs **control actions**, which **define driving strategy**
    
- Strategy emerges, not coded
    
- RL is acceptable **only** for tuning MPC parameters
    
- MPC remains the decision authority
    