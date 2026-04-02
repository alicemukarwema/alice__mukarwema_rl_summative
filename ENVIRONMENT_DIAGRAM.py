"""
Environment Architecture Diagram
ASCII representation of the Smart Farm Disease Management Environment
"""

# ==============================================================================
# ENVIRONMENT ARCHITECTURE DIAGRAM
# ==============================================================================

ENVIRONMENT_DIAGRAM = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SMART CROP DISEASE MANAGEMENT ENVIRONMENT                │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         5x5 FARM GRID                                │  │
│  │                                                                       │  │
│  │   ┌────┬────┬────┬────┬────┐                                        │  │
│  │   │    │    │    │    │    │  Health %, Disease %                  │  │
│  │   ├────┼────┼────┼────┼────┤  Colors:                              │  │
│  │   │    │ 🔴 │    │    │    │  🟢 Green = Healthy                  │  │
│  │   ├────┼────┼────┼────┼────┤  🔴 Red = Infected                   │  │
│  │   │    │ 🔴 │ 🔴 │    │    │  🟣 Dark Red = Severe                │  │
│  │   ├────┼────┼────┼────┼────┤  ⬜ Gray = Removed                    │  │
│  │   │    │ 🟢 │ 🔴 │    │    │                                       │  │
│  │   ├────┼────┼────┼────┼────┤  25 Plants Total                      │  │
│  │   │    │ 🟢 │ 🟢 │ 🟢 │    │  Initial: 1 infected plant           │  │
│  │   └────┴────┴────┴────┴────┘  Disease spreads over time            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGENT INTERACTION CYCLE                           │
│                                                                               │
│  ┌──────────┐                                                               │
│  │  OBSERVE │  ← Agent receives state observation                          │
│  │   STATE  │                                                               │
│  └────┬─────┘                                                               │
│       │                                                                      │
│       │  Observation = {                                                   │
│       │    crop_health: [85, 42, 78, ...],     # 25 values                │
│       │    disease_severity: [0, 45, 20, ...], # 25 values                │
│       │    soil_moisture: [65],                                           │
│       │    weather: [25°C, 65%, 35mm],         # temp, humidity, rain    │
│       │    days_since_infection: [0, 12, 8, ...],                        │
│       │    treatment_history: [3],                                        │
│       │    total_infected: [8]                                            │
│       │  }                                                                  │
│       ▼                                                                      │
│  ┌──────────┐                                                               │
│  │ DECIDE   │  ← Agent selects action based on policy                     │
│  │ ACTION   │                                                               │
│  └────┬─────┘                                                               │
│       │                                                                      │
│       │  Actions Available:                                                │
│       │  0: Monitor (no cost)                                              │
│       │  1: Spray Fungicide ($5, 70% effective)                           │
│       │  2: Apply Neem Oil ($2, 50% effective)                            │
│       │  3: Remove Plant ($10, prevents spread)                           │
│       │  4: Improve Ventilation ($1, reduce humidity)                     │
│       │  5: Adjust Irrigation ($0.50, optimal moisture)                   │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────┐                                                               │
│  │ EXECUTE  │  ← Environment applies action, updates state                │
│  │ ACTION   │                                                               │
│  └────┬─────┘                                                               │
│       │                                                                      │
│       │  Updates:                                                          │
│       │  - Apply disease reduction/prevention                             │
│       │  - Deduct treatment cost                                          │
│       │  - Update weather stochastically                                  │
│       │  - Reduce soil moisture                                           │
│       │  - Simulate disease spread to neighbors                           │
│       │  - Update plant health/disease values                             │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────┐                                                               │
│  │ CALCULATE│  ← Reward signal based on new state                         │
│  │ REWARD   │                                                               │
│  └────┬─────┘                                                               │
│       │                                                                      │
│       │  Reward = (Healthy Plants × 0.5)                                  │
│       │          + (Infected Plants × -2.0)                               │
│       │          - (Total Cost × 0.1)                                     │
│       │                                                                      │
│       │  Example: 20 healthy, 3 infected, $15 spent                       │
│       │  Reward = (20 × 0.5) + (3 × -2) - (15 × 0.1)                     │
│       │         = 10 - 6 - 1.5 = 2.5                                     │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────┐                                                               │
│  │ LEARN    │  ← Agent updates policy/value function                      │
│  │ POLICY   │                                                               │
│  └──────────┘                                                               │
│       │                                                                      │
│       └─────────────────────────────────────────────────────────┐          │
│                                                                   │          │
└───────────────────────────────────────────────────────────────┼──────────┘
                                                                  │
                                ┌─────────────────────────────────┘
                                │
                                ▼
                          [NEXT STEP/EPISODE]


┌─────────────────────────────────────────────────────────────────────────────┐
│                          STATE TRANSITION DIAGRAM                            │
│                                                                               │
│  Healthy Plant          ─Disease Exposure─→        Infected Plant           │
│  ┌──────────┐                                    ┌──────────┐              │
│  │ Health   │                                    │ Health   │              │
│  │  > 70%   │                                    │  30-70%  │              │
│  │ Disease  │                                    │ Disease  │              │
│  │   0%     │                                    │  30-60%  │              │
│  └──────────┘                                    └──────────┘              │
│       │                                                │                     │
│       │ - Cost: $0 (Monitor)                         │ Treatment:           │
│       │ - Health: Stable                             │ - Fungicide: -70%    │
│       │ - Spread: Low risk                           │ - Neem Oil: -50%     │
│       │ - Action: Monitor/Prevent                    │ - Remove: Eliminate  │
│       │                                               │ - Cost: $2-10        │
│       └─────────────────────────┬────────────────────┘                     │
│                                 │                                           │
│                                 ▼                                           │
│                     Severely Infected Plant                                 │
│                     ┌──────────┐                                            │
│                     │ Health   │                                            │
│                     │  < 30%   │                                            │
│                     │ Disease  │                                            │
│                     │  > 60%   │                                            │
│                     └──────────┘                                            │
│                     Spreads rapidly                                          │
│                     Must remove or                                           │
│                     lose entire patch                                        │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                       ENVIRONMENTAL FACTORS INTERACTION                      │
│                                                                               │
│                             Weather System                                   │
│                           ┌──────────────┐                                  │
│                           │ Temperature  │                                  │
│                           │ Humidity  ───┼─→ Disease Spread                 │
│                           │ Rainfall     │   (Affects Infection Rate)       │
│                           └──────────────┘                                  │
│                                                                               │
│                          Soil Moisture System                                │
│                          ┌──────────────┐                                   │
│                          │ Current Lvl  │                                   │
│                          │ Optimal: 70% ├─→ Health Impact                  │
│                          │ Too Low: +Stress                                 │
│                          │ Too High: -Infection │                           │
│                          └──────────────┘                                   │
│                                                                               │
│                        Treatment Cost Trade-off                              │
│                                                                               │
│  Immediate Cost              vs         Long-term Benefit                   │
│  ┌───────────────┐                      ┌──────────────┐                   │
│  │ Fungicide: $5 │ ───────────────────→ │ 70% Disease │                   │
│  │ Neem Oil: $2  │   Harvest Loss &     │ Reduction   │                   │
│  │ Remove: $10   │   Treatment Efficacy │              │                   │
│  │ Ventilate: $1 │                      │ vs          │                   │
│  │ Irrigate: $0.5│                      │ Total Yield │                   │
│  └───────────────┘                      └──────────────┘                   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                     AGENT LEARNING OBJECTIVE HIERARCHY                       │
│                                                                               │
│                          PRIMARY OBJECTIVE:                                  │
│                  "Minimize Disease | Maximize Yield"                        │
│                                │                                             │
│                ┌───────────────┼───────────────┐                            │
│                │               │               │                            │
│                ▼               ▼               ▼                            │
│         Early Detection   Efficient Treatment  Cost Control                 │
│         (Prevent spread)  (Right drug/dose)   (Budget aware)               │
│         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│         │ Monitor      │  │ Spray early  │  │ Prevent over │              │
│         │ Risk zones   │  │ to prevent   │  │ treatment    │              │
│         │ Daily checks │  │ spread       │  │ (cost $)     │              │
│         │ Act early    │  │ Remove only  │  │ Optimize $   │              │
│         │              │  │ if necessary │  │ per plant    │              │
│         └──────────────┘  └──────────────┘  └──────────────┘              │
│                │               │               │                            │
│                └───────────────┼───────────────┘                            │
│                                │                                             │
│                                ▼                                             │
│                   AGENT LEARNS OPTIMAL POLICY:                             │
│                  "When to treat, what to use"                              │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                     ALGORITHM LEARNING MECHANISMS                            │
│                                                                               │
│  DQN (Value-Based):            PPO (Policy Gradient):      A2C (Actor-Critic):│
│  ┌──────────────────┐         ┌──────────────────┐        ┌──────────────┐ │
│  │ Learns Q(s,a)    │         │ Learns π(a|s)    │        │ Learns π(a|s)│ │
│  │ - Value of each  │         │ - Direct policy  │        │ - Policy     │ │
│  │   action in each │         │ - Clipped updates│        │ + V(s) for   │ │
│  │   state          │         │ - Stable        │        │   variance   │ │
│  │ - Memorizes best │         │ - Efficient      │        │   reduction  │ │
│  │   Q-values       │         │                  │        │              │ │
│  │ - Greedy action  │         │                  │        │              │ │
│  │   selection      │         │                  │        │              │ │
│  └──────────────────┘         └──────────────────┘        └──────────────┘ │
│                                                                               │
│  Learns from:                Learns from:                Learns from:       │
│  - Past experiences          - Current trajectories      - Current + baseline│
│  - Replay buffer             - On-policy data            - Reduce variance   │
│  - Can use old policies      - Better final policy       - Simple, effective │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# ==============================================================================
# ASCII ART FARM VISUALIZATION
# ==============================================================================

FARM_VISUALIZATION = """
┌─────────────────────────────────────────────────────────────────┐
│             EXAMPLE FARM STATE SNAPSHOT (Step 25)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PLANT GRID (color shows health/disease):                      │
│                                                                  │
│      Col 0      Col 1      Col 2      Col 3      Col 4         │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐          │
│  │ 85% H   │ 78% H   │ 88% H   │ 92% H   │ 75% H   │ Row 0   │
│  │  0% D   │  2% D   │  0% D   │  0% D   │  3% D   │         │
│  ├─────────┼─────────┼─────────┼─────────┼─────────┤          │
│  │ 62% H   │ 45% H   │ 55% H   │ 80% H   │ 70% H   │ Row 1   │
│  │ 32% D   │ 42% D   │ 35% D   │  5% D   │ 12% D   │         │
│  ├─────────┼─────────┼─────────┼─────────┼─────────┤          │
│  │ 72% H   │ 40% H   │ 38% H   │ 85% H   │ 60% H   │ Row 2   │
│  │  8% D   │ 50% D   │ 52% D   │  2% D   │ 28% D   │         │
│  ├─────────┼─────────┼─────────┼─────────┼─────────┤          │
│  │ 90% H   │ 70% H   │ 42% H   │ 88% H   │ 82% H   │ Row 3   │
│  │  0% D   │ 18% D   │ 48% D   │  0% D   │  0% D   │         │
│  ├─────────┼─────────┼─────────┼─────────┼─────────┤          │
│  │ 76% H   │ 74% H   │ 35% H   │ 79% H   │ 81% H   │ Row 4   │
│  │  5% D   │  8% D   │ 55% D   │  6% D   │  1% D   │         │
│  └─────────┴─────────┴─────────┴─────────┴─────────┘          │
│                                                                  │
│  LEGEND:                                                        │
│  H = Health %  (0-100, plant vitality)                         │
│  D = Disease % (0-100, infection severity)                     │
│                                                                  │
│  STATUS SUMMARY:                                               │
│  ✓ Healthy Plants (H>70): 15 / 25                             │
│  ⚠ Infected Plants (D>30): 8 / 25                             │
│  ✗ Removed Plants: 0 / 25                                     │
│                                                                  │
│  ENVIRONMENT STATE:                                            │
│  Temperature: 24.5°C  |  Humidity: 62%  |  Rainfall: 28mm     │
│  Soil Moisture: 55%   |  Spread Risk: MEDIUM                  │
│                                                                  │
│  COST TRACKING:                                                │
│  Total Spent: $8.50                                            │
│  Last Action: Spray Fungicide (Cost: $5)                      │
│  Reward (Step): 2.35                                           │
│  Cumulative Reward: 42.18                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

# ==============================================================================
# DISEASE SPREAD MECHANISM
# ==============================================================================

DISEASE_SPREAD = """
┌─────────────────────────────────────────────────────────────────┐
│               DISEASE SPREAD MECHANISM (4-Connectivity)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Each step, infected plants (D > 30%) spread to neighbors:     │
│                                                                  │
│  Spread Probability = BASE_RATE × HUMIDITY × RAINFALL          │
│                    = 0.05 × (H/100) × (R/100)                  │
│                                                                  │
│  Example with Medium Humidity (65%) & Rain (40mm):             │
│  P(spread) = 0.05 × 0.65 × 0.40 = 0.013 = 1.3% per neighbor  │
│                                                                  │
│  4-Connectivity (spreads to adjacent, not diagonal):           │
│                                                                  │
│        ┌──────┐                                                │
│        │  UP  │                                                │
│        └──────┘                                                │
│           ▲                                                     │
│           │                                                     │
│  ┌──────┐─────┬──────┐                                         │
│  │ LEFT │PLANT│RIGHT │  Plant at (2,2) can infect:           │
│  └──────┴─────┴──────┘         (1,2) UP                        │
│           │                     (3,2) DOWN                      │
│           ▼                     (2,1) LEFT                      │
│        ┌──────┐                 (2,3) RIGHT                    │
│        │ DOWN │                 NOT diagonals                  │
│        └──────┘                                                │
│                                                                  │
│  If Infected Plant (2,2):                                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Each step, for each neighbor:                           │ │
│  │  1. Calculate spread probability                        │ │
│  │  2. Random number comparison                            │ │
│  │  3. If random < probability: spread disease             │ │
│  │  4. Add 5% disease severity to neighbor                 │ │
│  │                                                           │ │
│  │ Disease reduces plant health:                           │ │
│  │ Health_new = Health_old - 0.5 - (Disease × 0.01)       │ │
│  │                                                           │ │
│  │ Example: Health 85%, Disease 0% → Disease 5%            │ │
│  │ Health_new = 85 - 0.5 - (5 × 0.01) = 84.45%            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

# ==============================================================================
# EPISODE FLOW
# ==============================================================================

EPISODE_FLOW = """
┌─────────────────────────────────────────────────────────────────┐
│                 REINFORCEMENT LEARNING EPISODE FLOW             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  t=0: INITIALIZATION                                            │
│  ├─ Reset farm: All plants health 85%                         │
│  ├─ Initialize 1 infected plant (disease 40%, health 40%)      │
│  ├─ Set weather conditions                                     │
│  └─ Agent observes initial state                               │
│                                                                  │
│  t=1 to t=100: EPISODE LOOP                                    │
│  │                                                              │
│  ├─ STEP t:                                                    │
│  │  ├─ Agent observes state s_t                               │
│  │  ├─ Agent chooses action a_t from policy π(a|s)            │
│  │  ├─ Environment executes action                            │
│  │  │  ├─ Apply disease reduction/prevention                  │
│  │  │  ├─ Simulate disease spread to neighbors                │
│  │  │  ├─ Update plant health (decay due to disease)          │
│  │  │  ├─ Stochastic weather changes                          │
│  │  │  └─ Reduce soil moisture naturally                      │
│  │  ├─ Environment returns:                                   │
│  │  │  ├─ New state s_{t+1}                                   │
│  │  │  ├─ Reward r_t (immediate signal)                       │
│  │  │  ├─ Terminal/Truncated flag (episode end?)              │
│  │  │  └─ Info dict (diagnostics)                             │
│  │  └─ Agent updates policy (if learning)                     │
│  │                                                              │
│  ├─ Termination Check:                                         │
│  │  ├─ ALL PLANTS REMOVED/DEAD → Terminated (r -= 100)        │
│  │  ├─ t >= 100 → Truncated (max steps)                       │
│  │  └─ Otherwise → Continue                                   │
│  │                                                              │
│  └─ Repeat until terminated or truncated                       │
│                                                                  │
│  EPISODE END: Report                                            │
│  ├─ Total episode reward                                       │
│  ├─ Max healthy plants reached                                 │
│  ├─ Total cost spent                                           │
│  ├─ Final disease status                                       │
│  └─ Policy loss (if training)                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

def print_diagrams():
    """Print all diagrams"""
    print(ENVIRONMENT_DIAGRAM)
    print("\n")
    print(FARM_VISUALIZATION)
    print("\n")
    print(DISEASE_SPREAD)
    print("\n")
    print(EPISODE_FLOW)

if __name__ == "__main__":
    print_diagrams()
