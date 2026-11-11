# 🔋 NASA Li-ion Battery Degradation Dataset Column Explanation

### 🔹 `battery_id`

* Meaning: Experiment mein test kiye gaye individual lithium-ion cell ka unique identifier (jaise `B0005`, `B0006`, `B0007`, `B0018`).

* Values: Categorical string (`B0005`, `B0006`, etc.).

* Physical Significance: Har physical battery cell ki chemical composition aur internal manufacturing tolerance thodi alag hoti hai, isliye identical testing par bhi har cell ka degradation pattern thoda vary karta hai.
  
* Modeling Impact: Is column ko training ke dauran input feature **nahi** banana hai, varna model cell ID memorise kar lega. Iska primary use **Group-based Train/Test Split** ke liye hota hai (e.g., train on `B0005`/`B0006`, test on unseen `B0018`) taaki real-world generalization test ho sake.

---

### 🔹 `cycle`

* Meaning: Continuous charge-discharge cycle number. Battery lab mein kitni baar full charge ho kar discharge ho chuki hai.

* Expected Range: 1 se ~168+ cycles.


* Physical Significance: Battery ka operational "mileage". Har cycle ke sath anode par SEI (Solid Electrolyte Interphase) layer thick hoti jati hai aur active lithium ions permanently trap ho jate hain.

* Modeling Impact: Monotonic time-step feature. Trend analysis aur lag features (e.g., rolling rate of degradation) derive karne ke liye core reference index hai.

---

### 🔹 `voltage`

* Meaning: Us particular discharge cycle ke dauran record hua representative terminal voltage.

* Expected Range: ~3.53V – ~4.20V (discharge plateau levels ~3.53V se ~3.56V ke aas-paas).


* Physical Significance: Jaise-jaise cell age hota hai, internal resistance ($R_{int}$) badhti hai. High internal resistance ki wajah se load padte hi voltage drop ($IR$ drop) zyada tezi se hota hai aur battery cut-off voltage jaldi hit kar leti hai.

* Modeling Impact: Primary direct physical sensor feature. Voltage drop rate aur voltage stability battery wear-and-tear identify karne ke liye critical predictor hai.

---

### 🔹 `temperature`

* Meaning: Discharge cycle ke dauran cell surface par record hua ambient/operating temperature (°C mein).

* Expected Range: ~24°C – ~35°C (is dataset mein typical readings ~32.0°C – ~32.6°C ke around hain).


* Physical Significance: Degradation ke sath badhi hui internal resistance ki wajah se Joule heating ($I^2 R$) badh jati hai, jisse purani battery operating state mein fresh battery ke mukable zyada heat hoti hai.

* Modeling Impact: Core health indicator. Higher temperature rise rate ya anomalous heating directly accelerate degradation ko flag karti hai.

---

### 🔹 `capacity`

* Meaning: Us specific cycle mein battery dwara deliver ki gayi total usable electric charge capacity (Ampere-hours / Ah mein).

* Expected Range: ~1.86 Ah (fresh condition) se drop hokar ~1.40 Ah (degraded condition) tak. (Rated nominal fresh capacity = 2.0 Ah).

* Physical Significance: Battery health ka ground truth. Industry standard ke mutabiq jab capacity rated value ke 80% (approx 1.4 – 1.6 Ah) tak gir jati hai, battery ko End-of-Life (EOL) ghoshit karke critical applications se retire kar diya jata hai.

* Modeling Impact: Yeh ground-truth benchmark column hai jisse SOH aur RUL derive hote hain. Input feature ke roop mein directly drop kiya jata hai taaki direct data leakage na ho.

---

### 🔹 `soh` (State of Health)

* Meaning: Battery ki current health ratio, rated fresh capacity ke comparison mein normalized.

* Formula & Range: $\frac{\text{Current Capacity}}{\text{Initial Capacity}}$, value 1.0 (100% healthy) se lekar ~0.75 – 0.80 (20% degradation / End-of-Life).

* Physical Significance: EV dashboard par jo "Battery Health %" display hota hai, yeh wahi metric hai. 1.0 fresh condition represent karta hai.

* Modeling Impact: Intermediate target metric. Agar tumhara task SOH predict karna hai toh yeh target ($y$) banega; agar RUL predict kar rahe ho toh isse feature matrix se bahar rakha jata hai taaki target leakage prevent ho.

---

### 🔹 `rul` (Remaining Useful Life)

* Meaning: Current cycle se lekar battery ke fail / retirement threshold (80% SOH hit karne) tak bache hue cycles ki exact sankhya.

* Expected Range: ~167 cycles se countdown hokar 0 cycles tak.

* Physical Significance: Predictive maintenance ka final actionable output—"Yeh battery replacement ke bina aur kitne cycles reliably chalegi?"
 
* Modeling Impact: **Primary Target Variable ($y$)**. Regression models (LightGBM / CatBoost) is column ko predict karne ke liye optimize kiye jate hain.