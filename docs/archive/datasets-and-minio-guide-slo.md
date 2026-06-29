# ⚡ Smart Energy Data & MinIO Guide

> ⚠️ This document is written in Slovenian. See [smart-energy-data-description.md](../smart-energy-data-description.md) and [retraining-api-current-state.md](../retraining-api-current-state.md) for the authoritative English references.

Ta dokument razloži, kako v projektu **Smart Energy Active Learning** delamo s podatki, kateri dataseti so relevantni, kje so shranjeni na MinIO, od kod izvirajo in kako jih uporablja retraining API.

Glavni poudarek je trenutni dogovor s Kostasom: **velikega osnovnega dataseta ne prenašamo iz MinIO ob vsakem retrainu**. Osnovni dataset hranimo lokalno, iz MinIO pa ob vsakem klicu API-ja prenesemo samo majhno datoteko z novimi vrsticami.

---

## 🧭 Kratek povzetek

Trenutni produkcijski tok je:

```text
1. Digital Twin / dashboard ustvari nove N-1 označene vrstice
2. Nove vrstice se zapišejo na MinIO v appended_rows_latest.csv
3. Retraining API prenese samo ta majhen delta CSV
4. API lokalno združi:
   base dataset + appended rows
5. Random Forest se ponovno natrenira
6. Nov model in metrics.json se naložita nazaj na MinIO
```

Ključna odločitev:

```text
simulation_security_labels_n-1.csv
= velik osnovni dataset, prenese se lokalno enkrat

simulation_security_labels_n-1_appended_rows_latest.csv
= majhna delta datoteka, prenese se iz MinIO ob vsakem retrainu
```

---

## 🪣 MinIO lokacije

Glavni bucket:

```text
smart-energy-results
```

Relevantna mapa za Active Learning training podatke:

```text
smart-energy-results/al_training_dataset/
```

Trenutna struktura:

```text
smart-energy-results/
└── al_training_dataset/
    ├── simulation_security_labels_n-1.csv
    └── appended_rows/
        ├── simulation_security_labels_n-1_20260307T200307Z_20260110T030000.csv
        ├── simulation_security_labels_n-1_20260307T200515Z_20260110T040000.csv
        ├── ...
        └── simulation_security_labels_n-1_appended_rows_latest.csv
```

### 📌 Kaj pomeni posamezna lokacija?

| MinIO objekt | Namen | Kako ga uporabljamo |
|---|---|---|
| `al_training_dataset/simulation_security_labels_n-1.csv` | Osnovni labeled dataset za N-1 klasifikacijo | Prenese se lokalno enkrat in se ne prenaša ob vsakem API klicu |
| `al_training_dataset/appended_rows/*.csv` | Zgodovinske delta datoteke z novimi vrsticami | Uporabne za audit/debug, ne kot primarni input za retrain |
| `al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv` | Najnovejše nove vrstice, ki jih je treba dodati osnovnemu datasetu | To je primarni input za `/retrain` |
| `models/retraining_runs/<run_id>/model.joblib` | Natreniran Random Forest model | Output retraining API-ja |
| `models/retraining_runs/<run_id>/metrics.json` | Metrike zadnjega retraining runa | Output retraining API-ja |

---

## 📦 Relevantni lokalni dataseti v repozitoriju

V lokalnem repozitoriju so najpomembnejše datoteke:

```text
data/digital_twin_ext_grid.json
data/distributed_loads_uniform.csv
data/distributed_generators.csv
data/simulation_security_labels_n-1.csv
```

### 1. `data/digital_twin_ext_grid.json` 🕸️

To je statični **digitalni dvojček elektroenergetskega omrežja** v `pandapower` formatu.

Vsebuje:

- buse oziroma vozlišča,
- transmission lines,
- load elemente,
- generatorje,
- static generatorje,
- external grid oziroma slack referenco,
- tehnične parametre omrežja.

Ta datoteka ni neposreden ML training dataset. Je simulacijski model, ki omogoča izračun power flow in N-1 varnostnih preverjanj.

### 2. `data/distributed_loads_uniform.csv` 🔌

Ta datoteka vsebuje časovne vrste porabe oziroma load vrednosti.

Uporablja se pri pripravi operating pointov:

```text
timestamp + poraba po loadih -> vhod v digitalni dvojček
```

Ni končni supervised learning dataset, ampak eden izmed vhodnih virov za generiranje labeled dataseta.

### 3. `data/distributed_generators.csv` ⚙️

Ta datoteka vsebuje časovne vrste proizvodnje oziroma generator outpute.

Uporablja se skupaj z load podatki:

```text
timestamp + proizvodnja po generatorjih -> vhod v digitalni dvojček
```

Tudi ta datoteka ni neposredni retraining input, ampak vir za ustvarjanje operating pointov.

### 4. `data/simulation_security_labels_n-1.csv` ✅

To je glavni supervised ML dataset za **N-1 security classification**.

Vsaka vrstica pomeni en operating point omrežja pri določenem timestampu. Ciljna oznaka je:

```text
status = secure / insecure
```

Lokalno preverjeno stanje trenutne datoteke:

| Lastnost | Vrednost |
|---|---:|
| Število vrstic | 8,769 |
| Število stolpcev | 273 |
| `secure` vrstic | 4,497 |
| `insecure` vrstic | 4,272 |
| Velikost lokalne datoteke | približno 22.4 MB |

Glavne skupine stolpcev:

| Skupina | Primeri stolpcev | Pomen |
|---|---|---|
| Metadata | `timestamp` | Čas operating pointa |
| Target label | `status` | `secure` ali `insecure` |
| Base-case indikatorji | `max_line_loading_percent_basecase`, `min_bus_voltage_pu_basecase`, `max_bus_voltage_pu_basecase` | Stanje omrežja brez izpada |
| Contingency indikatorji | `max_line_loading_percent_contingency`, `min_bus_voltage_pu_contingency`, `max_bus_voltage_pu_contingency` | Najslabši rezultati N-1 preverjanja |
| Load features | `load_0_p_mw`, ..., `load_19_p_mw` | Poraba po load elementih |
| Generator features | `gen_0_p_mw`, ..., `gen_134_p_mw` | Proizvodnja po generatorjih |
| Static generator features | `sgen_0_p_mw`, ..., `sgen_109_p_mw` | Distributed generation / static generatorji |

---

## 🧪 Kaj pomeni `secure` in `insecure`?

`secure` pomeni, da operating point prestane N-1 varnostno preverjanje v digitalnem dvojčku.

`insecure` pomeni, da pri base-case ali N-1 contingency preverjanju pride do vsaj ene od težav:

- power-flow non-convergence,
- line loading nad dovoljeno mejo,
- bus voltage zunaj dovoljenega območja,
- druga simulacijska kršitev, ki jo labeling logika označi kot varnostno nesprejemljivo.

Pomembno:

```text
insecure != blackout
```

`insecure` pomeni, da stanje ne zadošča steady-state N-1 kriteriju v digitalnem dvojčku. To ni neposredna verjetnost izpada celotnega omrežja.

---

## 🔄 Dogovor s Kostasom glede MinIO workflowa

Prvotni pristop je bil, da bi ob retrainingu prenašali večji dataset iz MinIO. To je povzročalo timeout težave pri upload/download večjih datotek.

Zato je bil dogovor spremenjen:

### ❌ Ne delamo več tako

```text
API ob vsakem retrainu prenese celoten simulation_security_labels_n-1.csv iz MinIO
```

To je počasno in lahko povzroči timeout.

### ✅ Trenutni pravilni pristop

```text
1. simulation_security_labels_n-1.csv se lokalno pripravi/prenese enkrat
2. Dashboard/Digital Twin v MinIO zapiše samo nove vrstice
3. API ob retrainu prenese samo:
   al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv
4. API lokalno naredi:
   merged = base_dataset + appended_rows_latest
5. API natrenira model iz merged dataseta
```

Razlog:

- manjši prenos iz MinIO,
- manj timeoutov,
- hitrejši retraining,
- jasnejša ločitev med stabilnim base datasetom in novimi Active Learning vrsticami.

---

## 🧩 Kaj je `appended_rows_latest.csv`?

Datoteka:

```text
al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv
```

je **delta datoteka**, ne celoten dataset.

Vsebuje samo nove vrstice, ki jih je treba dodati osnovnemu datasetu.

Praktično:

```text
simulation_security_labels_n-1.csv
+ simulation_security_labels_n-1_appended_rows_latest.csv
= training dataset za trenutni retraining run
```

Zato mora imeti `appended_rows_latest.csv` enako shemo oziroma združljive stolpce kot base dataset. Dovoljeni so dodatni metadata stolpci, na primer:

```text
created_at
```

Retraining API trenutno privzeto odstrani:

```text
drop_latest_columns = ["created_at"]
```

---

## 🛠️ Kako `/retrain` uporablja podatke?

Endpoint:

```text
POST /retrain
```

Privzeti input iz MinIO:

```text
results_bucket = smart-energy-results
latest_key = al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv
```

Privzeta lokalna base datoteka v API kontejnerju:

```text
retraining-api/data/base/simulation_security_labels_n-1.csv
```

Lahko se prepiše z environment variable:

```text
BASE_DATASET_LOCAL_PATH=<path do base CSV>
```

Privzeta lokalna pot za prenesene appended rows:

```text
retraining-api/data/appended/simulation_security_labels_n-1_appended_rows_latest.csv
```

Lahko se prepiše z:

```text
APPENDED_ROWS_LOCAL_PATH=<path kamor se shrani delta CSV>
```

Minimalen request body:

```json
{}
```

Priporočen ekspliciten request body:

```json
{
  "results_bucket": "smart-energy-results",
  "latest_key": "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv",
  "output_prefix": "models/retraining_runs",
  "n_estimators": 400,
  "random_state": 42,
  "test_size": 0.2,
  "label_col": "status",
  "drop_feature_cols": [
    "timestamp",
    "max_line_loading_percent_basecase",
    "min_bus_voltage_pu_basecase",
    "max_bus_voltage_pu_basecase",
    "max_line_loading_percent_contingency",
    "min_bus_voltage_pu_contingency",
    "max_bus_voltage_pu_contingency"
  ],
  "drop_latest_columns": ["created_at"]
}
```

---

## 🚦 Kateri dataseti so relevantni?

### ✅ Relevantni za trenutno N-1 retraining integracijo

| Dataset | Relevantnost | Zakaj |
|---|---|---|
| `simulation_security_labels_n-1.csv` | Visoka | Glavni base dataset za treniranje klasifikatorja |
| `simulation_security_labels_n-1_appended_rows_latest.csv` | Visoka | Najnovejše Active Learning / dashboard dodane vrstice |
| `digital_twin_ext_grid.json` | Srednja/visoka | Vir simulacijske logike in razumevanja omrežja, ne neposreden retraining CSV |
| `distributed_loads_uniform.csv` | Srednja | Vir za generiranje operating pointov |
| `distributed_generators.csv` | Srednja | Vir za generiranje operating pointov |

### ⚠️ Relevantni samo za audit/debug

| Datoteka | Namen |
|---|---|
| `al_training_dataset/appended_rows/simulation_security_labels_n-1_<timestamp>_<timestamp>.csv` | Zgodovinski snapshoti novih vrstic |
| Lokalni prenosi iz `Downloads/` | Ročno preverjanje, primerjava sheme ali debug |
| Stari full snapshoti dataseta | Samo če želimo reproducirati star run |

### ❌ Ni primarni vir za trenutno N-1 retraining logiko

| Vir | Zakaj ni primarni |
|---|---|
| Day-ahead forecasting dataseti | To je ločen use case; XAI tam že obstaja, N-1 classification XAI še ni enako implementiran |
| Model output datoteke brez metrik | Model sam ni dataset in ne nadomešča labeled CSV |
| Dashboard prikaz verjetnosti | To so model predictions, ne ground-truth labels |

---

## 🧠 Od kod dobimo podatke?

Podatki niso ročno označeni s strani operaterja.

Tok nastanka podatkov:

```text
časovne vrste loadov
+ časovne vrste generatorjev
+ digitalni dvojček omrežja
+ base-case power flow
+ N-1 contingency simulations
= secure / insecure labeled dataset
```

Ground truth label prihaja iz:

```text
Digital Twin / N-1 simulation
```

Vloga operaterja/dashboarda v trenutnem Active Learning workflowu:

- dashboard identificira oziroma izbere negotove/informativne operating pointe,
- Digital Twin zanje izvede N-1 simulacijo,
- rezultat se zapiše kot nove labeled vrstice,
- API jih uporabi pri retrainingu.

Operater torej ne piše ročno label `secure` / `insecure`. Label nastane s simulacijo.

---

## 🧼 Pravila za varno delo s podatki

### ✅ Priporočeno

- Lokalni base dataset naj bo stabilen in verzioniran kot operativni input za retraining.
- Iz MinIO pri retrainingu prenašaj samo `appended_rows_latest.csv`.
- Pred retrainingom preveri, da ima appended file stolpce združljive z base datasetom.
- `created_at` ali podobne metadata stolpce odstrani pred treniranjem.
- Model in `metrics.json` vedno shrani v timestamped output mapo.
- Za paper in poročila uporabljaj samo dejanske metrike iz `metrics.json`.

### ❌ Ne priporočamo

- Ne prenašaj celotnega `simulation_security_labels_n-1.csv` iz MinIO pri vsakem API klicu.
- Ne obravnavaj `appended_rows_latest.csv` kot celoten dataset.
- Ne mešaj day-ahead forecasting XAI rezultatov z N-1 security classification rezultati.
- Ne spreminjaj base dataseta brez jasne evidence, katera verzija je bila uporabljena.
- Ne zapisuj model predictions kot ground-truth labels.

---

## 🔍 Minimalni validation checklist

Preden zaženemo retraining, je dobro preveriti:

- [ ] Ali lokalni `simulation_security_labels_n-1.csv` obstaja?
- [ ] Ali je MinIO objekt `simulation_security_labels_n-1_appended_rows_latest.csv` dostopen?
- [ ] Ali ima appended CSV stolpec `status`?
- [ ] Ali so vrednosti `status` samo `secure` / `insecure`?
- [ ] Ali so feature stolpci združljivi z base datasetom?
- [ ] Ali metadata stolpci, kot je `created_at`, ne pridejo v model features?
- [ ] Ali ima merged dataset dovolj vrstic obeh razredov za stratified train/test split?
- [ ] Ali se po retrainingu na MinIO naložita `model.joblib` in `metrics.json`?

---

## 🗂️ Priporočena dokumentacijska ločitev

Za projekt naj velja naslednja mentalna mapa:

```text
docs/archive/datasets-and-minio-guide-slo.md
= operativni vodič za podatke, MinIO in retraining dataset flow

reports/power_grid_dataset_learning_guide.md
= učni vodič za razumevanje elektroenergetskih pojmov in strukture dataseta

docs/retraining-api-current-state.md
= tehnično stanje FastAPI retraining implementacije
```

Ta dokument naj bo prvi vir za vprašanja:

- kateri dataset uporabiti,
- od kod pridejo nove vrstice,
- kaj je na MinIO,
- kaj mora prenesti API,
- kaj je base dataset in kaj je delta.

---

## 🧾 Kratek primer trenutnega retraining toka

```text
Base dataset:
data/simulation_security_labels_n-1.csv

MinIO delta:
smart-energy-results/al_training_dataset/appended_rows/
simulation_security_labels_n-1_appended_rows_latest.csv

API merge:
base + delta -> merged.csv

Training:
RandomForestClassifier

Output:
smart-energy-results/models/retraining_runs/<run_id>/model.joblib
smart-energy-results/models/retraining_runs/<run_id>/metrics.json
```

Najpomembnejša stvar:

> `simulation_security_labels_n-1_appended_rows_latest.csv` vsebuje samo nove vrstice. Ni zamenjava za osnovni dataset.

