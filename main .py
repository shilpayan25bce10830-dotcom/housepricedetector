
import os, json, pickle, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
for d in ["data","model","charts"]: os.makedirs(os.path.join(BASE,d), exist_ok=True)
CSV = os.path.join(BASE,"data","housing_data.csv")

FEATS = ["area_sqft","bedrooms","bathrooms","floors","age_years",
         "parking","location_enc","furnished_enc","road_access"]
C = {"bg":"#0d1117","sf":"#161b22","tx":"#e6edf3","mu":"#8b949e",
     "bl":"#58a6ff","gr":"#3fb950","ye":"#d29922","re":"#f85149","or":"#f0883e","pu":"#bc8cff"}

# ---------- scratch ML utilities ----------

def split(X, y, r=0.2, s=42):
    np.random.seed(s)
    i = np.random.permutation(len(X))
    c = int(len(X) * (1 - r))
    return X[i[:c]], X[i[c:]], y[i[:c]], y[i[c:]]

def scale(X):
    lo, hi = X.min(0), X.max(0)
    sp = hi - lo; sp[sp==0] = 1
    return (X-lo)/sp, lo, hi

def mae(a,b):  return np.mean(np.abs(a-b))
def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
def r2(a,b):   return 1 - np.sum((a-b)**2) / np.sum((a-np.mean(a))**2)

# ---------- Model 1: Linear Regression (gradient descent) ----------

class LinReg:
    def __init__(self, lr=0.01, epochs=2000):
        self.lr=lr; self.ep=epochs; self.w=None; self.b=0.
    def fit(self, X, y):
        n, f = X.shape; self.w=np.zeros(f); self.b=0.
        for _ in range(self.ep):
            e = X@self.w + self.b - y
            self.w -= self.lr*(2/n)*(X.T@e)
            self.b -= self.lr*(2/n)*e.sum()
    def predict(self, X): return X@self.w + self.b

# ---------- Model 2: Decision Tree Regressor ----------

class DTReg:
    def __init__(self, max_depth=8, min_samples=4):
        self.d=max_depth; self.ms=min_samples; self.tree=None
    def _vr(self, p, l, r):
        if len(l)==0 or len(r)==0: return 0
        n=len(p)
        return np.var(p)-(len(l)/n)*np.var(l)-(len(r)/n)*np.var(r)
    def _split(self, X, y):
        bs,bf,bt = -1,None,None
        for f in range(X.shape[1]):
            for t in np.unique(X[:,f]):
                s = self._vr(y, y[X[:,f]<=t], y[X[:,f]>t])
                if s>bs: bs,bf,bt = s,f,t
        return bf,bt
    def _build(self, X, y, d):
        if d==0 or len(y)<=self.ms: return {"leaf":True, "val":float(np.mean(y))}
        f,t = self._split(X,y)
        if f is None: return {"leaf":True, "val":float(np.mean(y))}
        m = X[:,f]<=t
        if m.sum()==0 or (~m).sum()==0: return {"leaf":True, "val":float(np.mean(y))}
        return {"leaf":False,"f":f,"t":t,
                "L":self._build(X[m],y[m],d-1),
                "R":self._build(X[~m],y[~m],d-1)}
    def fit(self, X, y):   self.tree = self._build(X, y, self.d)
    def _p1(self, x, n):   return n["val"] if n["leaf"] else self._p1(x, n["L"] if x[n["f"]]<=n["t"] else n["R"])
    def predict(self, X):  return np.array([self._p1(r,self.tree) for r in X])

# ---------- Model 3: Random Forest (bagging of decision trees) ----------

class RandForest:
    def __init__(self, n=30, max_depth=7):
        self.n=n; self.d=max_depth; self.trees=[]
    def fit(self, X, y):
        self.trees=[]
        np.random.seed(99)
        for _ in range(self.n):
            idx = np.random.choice(len(X), len(X), replace=True)
            t = DTReg(max_depth=self.d)
            t.fit(X[idx], y[idx])
            self.trees.append(t)
    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees])
        return preds.mean(axis=0)

# ---------- STEP 1: Create dataset ----------

def make_data():
    np.random.seed(42)
    n = 800
    location_map  = {"Rural":0, "Suburban":1, "Urban":2, "Prime":3}
    furnished_map = {"Unfurnished":0, "Semi":1, "Furnished":2}

    area    = np.random.randint(400, 4500, n).astype(float)
    beds    = np.random.choice([1,2,3,4,5], n, p=[0.05,0.25,0.40,0.20,0.10]).astype(float)
    baths   = np.clip(beds - np.random.choice([0,1], n, p=[0.4,0.6]), 1, 4).astype(float)
    floors  = np.random.choice([1,2,3], n, p=[0.50,0.35,0.15]).astype(float)
    age     = np.random.randint(0, 40, n).astype(float)
    parking = np.random.choice([0,1,2], n, p=[0.20,0.55,0.25]).astype(float)
    loc_enc = np.random.choice([0,1,2,3], n, p=[0.15,0.30,0.35,0.20]).astype(float)
    fur_enc = np.random.choice([0,1,2], n, p=[0.25,0.40,0.35]).astype(float)
    road    = np.random.choice([0,1], n, p=[0.30,0.70]).astype(float)

    # Price formula: area is the biggest driver, location multiplies it
    loc_mult = {0:1.0, 1:1.3, 2:1.8, 3:2.5}
    base_price = area * 3200
    price = (base_price
             * np.array([loc_mult[l] for l in loc_enc.astype(int)])
             + beds    * 150000
             + baths   * 80000
             + floors  * 60000
             - age     * 25000
             + parking * 90000
             + fur_enc * 120000
             + road    * 100000
             + np.random.normal(0, 200000, n))
    price = np.clip(price, 400000, 25000000).round(-3)

    df = pd.DataFrame({
        "area_sqft":    area,
        "bedrooms":     beds,
        "bathrooms":    baths,
        "floors":       floors,
        "age_years":    age,
        "parking":      parking,
        "location_enc": loc_enc,
        "furnished_enc":fur_enc,
        "road_access":  road,
        "price":        price,
        "location_name":  [["Rural","Suburban","Urban","Prime"][int(l)] for l in loc_enc],
        "furnished_name": [["Unfurnished","Semi","Furnished"][int(f)] for f in fur_enc],
    })
    df.to_csv(CSV, index=False)
    print(f"  Dataset created: {len(df)} records  ->  {CSV}")
    return df

def load_data():
    return pd.read_csv(CSV) if os.path.exists(CSV) else make_data()

# ---------- STEP 2: EDA charts (single page, 6 panels) ----------

def show_charts(df):
    print(f"\nStep 2 | EDA")
    print(f"  Records  : {len(df)}")
    print(f"  Avg price: Rs {df['price'].mean():,.0f}")
    print(f"  Min price: Rs {df['price'].min():,.0f}")
    print(f"  Max price: Rs {df['price'].max():,.0f}")

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16,9), facecolor=C["bg"])
    fig.suptitle("Housing Price Predictor  |  Data Analysis",
                 color=C["tx"], fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Price distribution
    ax = fig.add_subplot(gs[0,0])
    ax.hist(df["price"]/1e6, bins=30, color=C["bl"], edgecolor=C["bg"], alpha=0.9)
    ax.set_facecolor(C["sf"])
    ax.set_title("Price Distribution", color=C["tx"])
    ax.set_xlabel("Price (Rs lakhs)", color=C["mu"])
    ax.set_ylabel("Count", color=C["mu"])
    ax.tick_params(colors=C["mu"])

    # 2. Area vs price scatter
    ax = fig.add_subplot(gs[0,1])
    sc = ax.scatter(df["area_sqft"], df["price"]/1e6, c=df["location_enc"],
                    cmap="cool", alpha=0.5, s=12)
    ax.set_facecolor(C["sf"])
    ax.set_title("Area vs Price (colour = location)", color=C["tx"])
    ax.set_xlabel("Area (sqft)", color=C["mu"])
    ax.set_ylabel("Price (Rs lakhs)", color=C["mu"])
    ax.tick_params(colors=C["mu"])
    plt.colorbar(sc, ax=ax, shrink=0.8).ax.tick_params(colors=C["mu"])

    # 3. Avg price by location
    ax = fig.add_subplot(gs[0,2])
    loc_avg = df.groupby("location_name")["price"].mean().sort_values()
    bars = ax.barh(loc_avg.index, loc_avg.values/1e6,
                   color=[C["gr"],C["bl"],C["ye"],C["re"]], edgecolor=C["bg"], height=0.6)
    ax.set_facecolor(C["sf"])
    ax.set_title("Avg Price by Location", color=C["tx"])
    ax.set_xlabel("Rs lakhs", color=C["mu"])
    ax.tick_params(colors=C["mu"])
    for bar, v in zip(bars, loc_avg.values/1e6):
        ax.text(v+0.1, bar.get_y()+bar.get_height()/2, f"{v:.0f}L",
                va="center", color=C["tx"], fontsize=8)

    # 4. Avg price by bedrooms
    ax = fig.add_subplot(gs[1,0])
    bed_avg = df.groupby("bedrooms")["price"].mean()
    bars = ax.bar(bed_avg.index.astype(int), bed_avg.values/1e6,
                  color=C["pu"], edgecolor=C["bg"], width=0.6)
    ax.set_facecolor(C["sf"])
    ax.set_title("Avg Price by Bedrooms", color=C["tx"])
    ax.set_xlabel("Bedrooms", color=C["mu"])
    ax.set_ylabel("Rs lakhs", color=C["mu"])
    ax.tick_params(colors=C["mu"])
    for bar, v in zip(bars, bed_avg.values/1e6):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{v:.0f}L", ha="center", color=C["tx"], fontsize=8)

    # 5. Price vs house age
    ax = fig.add_subplot(gs[1,1])
    ax.scatter(df["age_years"], df["price"]/1e6, c=C["or"], alpha=0.4, s=10)
    # trend line using simple linear fit
    z = np.polyfit(df["age_years"], df["price"]/1e6, 1)
    xline = np.linspace(df["age_years"].min(), df["age_years"].max(), 100)
    ax.plot(xline, np.polyval(z, xline), color=C["re"], linewidth=2, linestyle="--")
    ax.set_facecolor(C["sf"])
    ax.set_title("Age vs Price", color=C["tx"])
    ax.set_xlabel("House Age (years)", color=C["mu"])
    ax.set_ylabel("Rs lakhs", color=C["mu"])
    ax.tick_params(colors=C["mu"])

    # 6. Furnished vs price boxplot
    ax = fig.add_subplot(gs[1,2])
    labels = ["Unfurnished","Semi","Furnished"]
    bp = ax.boxplot([df[df["furnished_name"]==l]["price"].values/1e6 for l in labels],
                    patch_artist=True, medianprops={"color":C["tx"],"linewidth":2})
    for box, col in zip(bp["boxes"], [C["mu"],C["bl"],C["gr"]]):
        box.set_facecolor(col); box.set_alpha(0.8)
    ax.set_facecolor(C["sf"])
    ax.set_title("Price by Furnished Status", color=C["tx"])
    ax.set_xticklabels(labels, color=C["mu"], fontsize=8)
    ax.tick_params(axis="y", colors=C["mu"])
    ax.set_ylabel("Rs lakhs", color=C["mu"])

    out = os.path.join(BASE,"charts","analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"  Chart saved -> {out}")
    plt.show(); plt.close()

# ---------- STEP 3 + 4 + 5: Feature engineering, training, evaluation ----------

def run_pipeline(df):
    X = df[FEATS].values.astype(float)
    y = df["price"].values.astype(float)
    Xs, xlo, xhi = scale(X)
    Xtr, Xte, ytr, yte = split(X, y)
    Xstr, Xste, _, _   = split(Xs, y)

    print("\nStep 3 | Features:", FEATS)
    print(f"\nStep 4 | Training   Train:{len(Xtr)}  Test:{len(Xte)}")
    print("  ┌────────────────────────┬──────────────┬──────────────┬───────┐")
    print("  │ Model                  │     MAE      │     RMSE     │   R2  │")
    print("  ├────────────────────────┼──────────────┼──────────────┼───────┤")

    lr = LinReg(lr=0.01, epochs=2000);  lr.fit(Xstr, ytr);  lp = lr.predict(Xste)
    dt = DTReg(max_depth=8);             dt.fit(Xtr,  ytr);  dp = dt.predict(Xte)
    rf = RandForest(n=30, max_depth=7);  rf.fit(Xtr,  ytr);  rp = rf.predict(Xte)

    results = [
        ("Linear Regression", lr, lp, True),
        ("Decision Tree",     dt, dp, False),
        ("Random Forest  ★",  rf, rp, False),
    ]
    for name, _, p, _ in results:
        print(f"  │  {name:<23}│ Rs {mae(yte,p)/1e5:7.1f}L  │ Rs {rmse(yte,p)/1e5:7.1f}L  │ {r2(yte,p):5.2f} │")
    print("  └────────────────────────┴──────────────┴──────────────┴───────┘")

    best_name, best_model, best_preds, best_scaled = max(results, key=lambda x: r2(yte, x[2]))
    print(f"  Best: {best_name.strip()}  R2={r2(yte,best_preds):.2f}")

    # Evaluation chart
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12,5), facecolor=C["bg"])
    fig.suptitle("Housing Predictor — Model Evaluation", color=C["tx"], fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(yte/1e6, best_preds/1e6, c=C["bl"], alpha=0.5, s=12)
    mn, mx = min(yte.min(), best_preds.min())/1e6, max(yte.max(), best_preds.max())/1e6
    ax.plot([mn,mx],[mn,mx], color=C["re"], linewidth=1.5, linestyle="--")
    ax.set_facecolor(C["sf"]); ax.set_title("Actual vs Predicted Price", color=C["tx"])
    ax.set_xlabel("Actual (Rs lakhs)", color=C["mu"]); ax.set_ylabel("Predicted (Rs lakhs)", color=C["mu"])
    ax.tick_params(colors=C["mu"])

    ax = axes[1]
    residuals = (yte - best_preds)/1e6
    ax.hist(residuals, bins=30, color=C["gr"], edgecolor=C["bg"], alpha=0.85)
    ax.axvline(0, color=C["re"], linewidth=1.5, linestyle="--")
    ax.set_facecolor(C["sf"]); ax.set_title("Prediction Residuals", color=C["tx"])
    ax.set_xlabel("Actual - Predicted (Rs lakhs)", color=C["mu"]); ax.tick_params(colors=C["mu"])

    plt.tight_layout()
    out = os.path.join(BASE,"charts","evaluation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"\nStep 5 | Eval chart saved -> {out}")
    plt.show(); plt.close()

    art = {"model":best_model, "scaled":best_scaled, "xlo":xlo, "xhi":xhi,
           "features":FEATS, "name":best_name.strip()}
    pickle.dump(art, open(os.path.join(BASE,"model","model.pkl"),"wb"))
    json.dump({"model":best_name.strip(),"r2":round(r2(yte,best_preds),3),
               "rmse":round(rmse(yte,best_preds)/1e5,2),"size":len(df)},
              open(os.path.join(BASE,"model","meta.json"),"w"), indent=2)
    print(f"  Model saved -> model/model.pkl")
    return art

# ---------- STEP 6: Interactive predictor ----------

def predictor(art=None):
    if art is None:
        try: art = pickle.load(open(os.path.join(BASE,"model","model.pkl"),"rb"))
        except: print("Run full pipeline first."); return

    m = art["model"]
    print("\n" + "="*48)
    print("  Housing Price Predictor")
    print("  Enter house details to get an estimated price.")
    print("  Type 'q' to quit.")
    print("="*48)

    while True:
        print("\n" + "─"*48)
        try:
            area  = float(input("  Area in sqft (e.g. 1200)           : "))
            if str(area).lower() == "q": break
            beds  = int(input(  "  Bedrooms (1-5)                     : "))
            baths = int(input(  "  Bathrooms (1-4)                    : "))
            flrs  = int(input(  "  Number of floors (1-3)             : "))
            age   = int(input(  "  Age of house in years              : "))
            park  = int(input(  "  Parking spots (0/1/2)              : "))
            print("  Location: 0=Rural  1=Suburban  2=Urban  3=Prime")
            loc   = int(input(  "  Location type (0-3)                : "))
            print("  Furnished: 0=Unfurnished  1=Semi  2=Fully furnished")
            fur   = int(input(  "  Furnished status (0-2)             : "))
            road  = int(input(  "  Main road access? (1=Yes / 0=No)  : "))
        except (ValueError, KeyboardInterrupt):
            print("  Invalid input, please try again."); continue

        Xi = np.array([[area, beds, baths, flrs, age, park, loc, fur, road]], float)
        if art["scaled"]:
            sp = art["xhi"] - art["xlo"]; sp[sp==0]=1
            Xi = (Xi - art["xlo"]) / sp
        price = m.predict(Xi)[0]
        price = max(400000, price)

        loc_names = {0:"Rural", 1:"Suburban", 2:"Urban", 3:"Prime"}
        fur_names  = {0:"Unfurnished", 1:"Semi-furnished", 2:"Fully furnished"}

        print(f"""
  ┌──────────────────────────────────────────┐
  │         PRICE ESTIMATE                   │
  ├──────────────────────────────────────────┤
  │  Area       : {area:.0f} sqft                    
  │  Bedrooms   : {beds}   Bathrooms: {baths}              
  │  Location   : {loc_names.get(loc,"?")}                    
  │  Furnished  : {fur_names.get(fur,"?")}          
  │  Age        : {age} years                     
  ├──────────────────────────────────────────┤
  │  Estimated Price : Rs {price/1e6:.2f} Lakhs       
  │                  = Rs {price:,.0f}         
  └──────────────────────────────────────────┘""")

        price_range_low  = price * 0.90
        price_range_high = price * 1.10
        print(f"  Likely range: Rs {price_range_low/1e6:.2f}L  —  Rs {price_range_high/1e6:.2f}L")

        if loc <= 1:
            print("  Tip: Urban/Prime location could increase price by 50-150%.")
        if age > 20:
            print("  Tip: House is older — price may drop further with age.")
        if fur == 0:
            print("  Tip: Furnishing the house could add Rs 1-3 Lakhs to value.")

        if input("\n  Predict another house? (y/n): ").strip().lower() != "y":
            print("\n  Happy house hunting!\n"); break

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--eda",     action="store_true")
    args = ap.parse_args()

    if args.predict: predictor(); return

    print("="*48)
    print("  Housing Price Predictor")
    print("  Pure numpy + pandas  |  No sklearn")
    print("="*48)

    print("\nStep 1 | Loading data")
    df = load_data()
    print(f"  {len(df)} records loaded.")

    show_charts(df)
    if args.eda: return

    art = run_pipeline(df)
    if input("\n  Try a prediction? (y/n): ").strip().lower() == "y":
        predictor(art)

    print("\nDone!")
    print("  data/housing_data.csv  |  model/model.pkl  |  charts/\n")

if __name__ == "__main__": main()
