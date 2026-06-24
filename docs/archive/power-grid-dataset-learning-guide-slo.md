# Ucni vodic po Smart Energy datasetu in elektroenergetskem omrezju

> ⚠️ This document is written in Slovenian. It is a personal learning resource on power-grid concepts. See [smart-energy-data-description.md](../smart-energy-data-description.md) for the authoritative English dataset reference.

Ta dokument razlozi, kaj vsebujejo podatki v projektu Smart Energy, iz cesa je sestavljen digitalni dvojcek omrezja in kako iz osnovnih elektroenergetskih pojmov pridemo do N-1 security classification dataseta.

Namenjen je bralcu, ki se sele uci pojme, kot so bus, generator, load, line, transformer, MW, napetost, tok, pretoki moci, digitalni dvojcek in N-1 kriterij.

Dokument je pisan kot prakticna razlaga za projektne datoteke:

- `data/digital_twin_ext_grid.json`
- `data/distributed_loads_uniform.csv`
- `data/distributed_generators.csv`
- `data/simulation_security_labels_n-1.csv`
- `notebooks/prepare_security_classification_dataset.ipynb`
- `notebooks/train_random_forest_security_classifier.ipynb`
- `models/random_forest_model.pkl`

---

## 1. Kaj dataset dejansko predstavlja

Dataset predstavlja urne operating pointe elektroenergetskega prenosnega omrezja in oznako, ali je celotno stanje omrezja po N-1 kriteriju `secure` ali `insecure`.

Najbolj enostavno:

```text
en timestamp
+ poraba po vozliscih
+ proizvodnja po generatorjih
+ digitalni model omrezja
+ power-flow simulacija
+ N-1 contingency simulacije
= secure / insecure label
```

V tem projektu ne klasificiramo samo ene linije ali enega generatorja. Klasificiramo stanje celotnega omrezja pri dani uri.

Vsaka vrstica v koncnem datasetu pomeni priblizno:

```text
Ali je to elektroenergetsko omrezje pri tej porabi in tej proizvodnji varno,
ce odpove en element, na primer ena linija ali en generator?
```

Pomembno: labela ni rocno dodeljena s strani operaterja. Labela `secure` / `insecure` nastane s simulacijo v digitalnem dvojcku.

---

## 2. Kaj je digitalni dvojcek omrezja

Digitalni dvojcek je racunalniski model realnega ali realisticnega fizicnega sistema.

V tem projektu je digitalni dvojcek elektroenergetsko omrezje, zapisano v `pandapower` formatu:

```text
data/digital_twin_ext_grid.json
```

To ni samo tabela podatkov. To je simulacijski model, ki vsebuje:

- vozlisca oziroma buse,
- povezave oziroma transmission lines,
- generatorje,
- static generatorje oziroma distributed generation,
- porabnike oziroma loads,
- external grid oziroma slack bus,
- tehnicne parametre linij,
- nazivne napetosti,
- dovoljene napetostne meje,
- tokovne omejitve linij,
- topologijo povezav med elementi.

S taksnim modelom lahko racunamo power flow: to pomeni, da za dano porabo in proizvodnjo izracunamo napetosti po vozliscih in pretoke po linijah.

---

## 3. Konkretna sestava digitalnega dvojcka

Iz `digital_twin_ext_grid.json` je trenutni model sestavljen iz naslednjih glavnih elementov:

| Element v pandapower | Stevilo vrstic | Pomen |
|---|---:|---|
| `bus` | 35 | Vozlisca omrezja |
| `line` | 46 | Prenosne linije |
| `load` | 20 | Porabniska vozlisca oziroma odjemi |
| `gen` | 135 | Generatorji |
| `sgen` | 110 | Static generatorji / distributed generation |
| `ext_grid` | 1 | External grid oziroma slack referenca |
| `trafo` | 0 | Tabela za transformatorje obstaja, vendar v tem modelu nima elementov |
| `trafo3w` | 0 | Tabela za trikrake transformatorje obstaja, vendar v tem modelu nima elementov |
| `switch` | 0 | Brez eksplicitnih stikal v trenutnem modelu |
| `storage` | 0 | Brez eksplicitnih hranilnikov energije |

Vse bus napetosti v modelu so na ravni:

```text
vn_kv = 380 kV
```

To pomeni, da je model v tej verziji osredotocen na visokonapetostno prenosno omrezje.

Napetostne meje, zapisane pri busih v digitalnem dvojcku, so:

```text
min_vm_pu = 0.95
max_vm_pu = 1.05
```

V sami N-1 labeling kodi pa se za varnostni check uporabljajo sirse operativne meje:

```text
0.9 <= vm_pu <= 1.1
```

To pomeni:

- podatkovni model busov vsebuje nominalne/mejne nastavitve 0.95-1.05 pu,
- notebook za security label uporablja kriterij 0.9-1.1 pu.

To razliko je dobro imeti v mislih, ker ena vrednost opisuje parametre modela, druga pa trenutno uporabljeno pravilo za oznacevanje.

---

## 4. Kaj je bus

Bus je vozlisce elektroenergetskega omrezja.

V fizicnem svetu si bus lahko predstavljamo kot zbiralko v razdelilni ali transformatorski postaji. To je tocka, kjer se lahko srecajo:

- linije,
- generatorji,
- porabniki,
- transformatorji,
- zunanji priklop omrezja.

V racunalniskem modelu je bus osnovna tocka topologije. Linija ima vedno zacetni in koncni bus:

```text
line = povezava med from_bus in to_bus
```

V trenutnem digitalnem dvojcku je 35 busov. Primeri imen busov so:

```text
1207, 1208, 1209, 1210, ..., 1299, 2074, 8173, 8446, ..., Slack
```

Vsak bus ima nazivno napetost:

```text
vn_kv = 380 kV
```

To ne pomeni, da je dejanska napetost vedno tocno 380 kV. Pomeni, da je to nominalna napetostna raven. V simulaciji se dejanska napetost pogosto izraza v `pu`.

---

## 5. Kaj pomeni pu

`pu` pomeni per-unit.

Per-unit sistem izrazi vrednost relativno glede na nazivno oziroma bazno vrednost.

Za napetost:

```text
V_pu = V_actual / V_nominal
```

Ce je nazivna napetost 380 kV:

```text
V_pu = 1.0
```

pomeni priblizno 380 kV.

```text
V_pu = 0.95
```

pomeni priblizno 95 % nazivne napetosti.

```text
V_pu = 1.05
```

pomeni priblizno 105 % nazivne napetosti.

Zakaj se uporablja `pu`? Ker je v velikih omrezjih lazje primerjati napetostne odklone relativno, namesto da stalno delamo z absolutnimi kV vrednostmi.

---

## 6. Kaj je line

Line je elektricna povezava med dvema busoma. V prenosnem omrezju je to obicajno daljnovod oziroma visokonapetostna prenosna povezava.

V trenutnem digitalnem dvojcku je:

```text
46 linij
```

Vse linije imajo isti standardni tip:

```text
Al/St 240/40 4-bundle 380.0
```

Vsaka linija ima parametre, kot so:

- `from_bus`: zacetno vozlisce,
- `to_bus`: koncno vozlisce,
- `length_km`: dolzina linije,
- `r_ohm_per_km`: upornost na kilometer,
- `x_ohm_per_km`: reaktanca na kilometer,
- `c_nf_per_km`: kapacitivnost na kilometer,
- `max_i_ka`: maksimalni dovoljeni tok,
- `in_service`: ali je linija aktivna v simulaciji.

V tem modelu imajo linije:

```text
max_i_ka = 2.0 kA
```

Dolzine linij so priblizno:

```text
min = 5.40 km
max = 155.85 km
average = 56.98 km
```

Primeri povezav:

| Linija | From bus | To bus | Dolzina |
|---|---|---|---:|
| `4617` | `1207` | `1209` | 28.04 km |
| `10950` | `1208` | `1209` | 21.51 km |
| `3601` | `1208` | `1210` | 102.61 km |
| `10949` | `1208` | `1210` | 97.44 km |
| `5074` | `1212` | `1213` | 47.89 km |

Obstaja tudi posebna linija:

```text
line_45 = Slack-to-Bus, Slack -> 1210, length = 30 km
```

Ta linija se v notebooku izkljuci iz N-1 contingency seznama in iz nekaterih line-loading izracunov:

```text
contingency_lines = [idx for idx in net.line.index if idx != 45]
```

To je pomembno, ker pomeni, da se N-1 simulacije izvajajo nad ostalimi linijami, ne nad povezavo `Slack-to-Bus`.

---

## 7. Kaj je load

Load pomeni porabnik oziroma odjem elektricne energije.

V fizicnem svetu je load lahko:

- mesto,
- industrijska cona,
- vecje regionalno odjemno obmocje,
- agregirana poraba vec uporabnikov.

V tem modelu je:

```text
20 load elementov
```

Vsak load je prikljucen na nek bus. V `digital_twin_ext_grid.json` so zacetne vrednosti `p_mw` nastavljene na 0, nato pa jih notebook pri vsaki uri napolni iz:

```text
data/distributed_loads_uniform.csv
```

Primer:

```text
load_0, load_1, ..., load_19
```

V koncnem security datasetu se ti stolpci pojavijo kot:

```text
load_0_p_mw, load_1_p_mw, ..., load_19_p_mw
```

To pomeni aktivno moc porabe posameznega load elementa v MW.

---

## 8. Kaj je generator

Generator je element, ki proizvaja elektricno moc in jo oddaja v omrezje.

V `pandapower` modelu je `gen` tipicno napetostno reguliran generator. To pomeni, da generator ne predstavlja samo vira aktivne moci, ampak lahko sodeluje tudi pri vzdrzevanju napetosti na svojem busu.

V trenutnem modelu je:

```text
135 gen elementov
```

Primeri imen generatorjev:

```text
C12_lignite
C13_lignite
C14_OCGT
C15_OCGT
C35_nuclear
C38_CCGT
C39_coal
```

Ime pogosto nakazuje tip proizvodnje:

- `lignite`: lignit,
- `OCGT`: open-cycle gas turbine,
- `CCGT`: combined-cycle gas turbine,
- `coal`: premog,
- `oil`: olje,
- `nuclear`: jedrska proizvodnja.

V casovni datoteki `distributed_generators.csv` so stolpci:

```text
gen_0, gen_1, ..., gen_134
```

V koncnem ML datasetu so zapisani kot:

```text
gen_0_p_mw, gen_1_p_mw, ..., gen_134_p_mw
```

---

## 9. Kaj je static generator oziroma sgen

`sgen` pomeni static generator.

V `pandapower` se `sgen` pogosto uporablja za vire, ki jih modeliramo kot injekcijo moci v bus, brez enake napetostne regulacijske vloge kot klasicni `gen`.

V praksi lahko `sgen` predstavlja distribuirano ali obnovljivo proizvodnjo, na primer:

- soncne elektrarne,
- vetrne elektrarne,
- biomasa,
- geotermalna proizvodnja,
- manjse razprsene enote.

V trenutnem modelu je:

```text
110 sgen elementov
```

Primeri imen:

```text
C41_geothermal
C42_biomass
C49_geothermal
C50_biomass
```

V casovni datoteki `distributed_generators.csv` so poleg `gen_*` tudi stolpci:

```text
sgen_0, sgen_1, ..., sgen_109
```

V koncnem datasetu se pojavijo kot:

```text
sgen_0_p_mw, sgen_1_p_mw, ..., sgen_109_p_mw
```

---

## 10. Kaj je external grid oziroma slack bus

`ext_grid` predstavlja zunanjo omrezno referenco. V power-flow izracunu je to pogosto slack bus.

Slack bus ima posebno nalogo:

- doloci referencni kot napetosti,
- pokrije razliko med proizvodnjo, porabo in izgubami,
- omogoci, da ima power-flow izracun matematicno referenco.

V trenutnem modelu je:

```text
1 external grid
```

Ime:

```text
Slack
```

Prikljucen je na bus `Slack`, ki je prek linije `Slack-to-Bus` povezan z omrezjem.

Pomembno: slack bus ni navaden generator v smislu ucnega dataseta. Je referencni element za simulacijo.

---

## 11. Ali imamo transformatorje

V splosnem elektroenergetskem omrezju so transformatorji zelo pomembni. Transformator povezuje razlicne napetostne nivoje, na primer:

```text
400 kV -> 220 kV
220 kV -> 110 kV
110 kV -> 20 kV
```

Osnovna naloga transformatorja je pretvorba napetostnega nivoja.

V tem konkretnem `pandapower` digitalnem dvojcku tabele za transformatorje obstajajo, vendar nimajo elementov:

```text
trafo rows = 0
trafo3w rows = 0
```

To pomeni:

- transformatorji so del `pandapower` sheme,
- vendar v tej verziji modela niso eksplicitno modelirani kot `trafo` ali `trafo3w` elementi,
- celoten opazovani model je na 380 kV napetostnem nivoju.

Zato je pravilno reci:

```text
Projektno omrezje v tej datoteki vsebuje bus-e, linije, loads, gen, sgen in ext_grid.
Eksplicitnih transformatorjev v trenutnem digitalnem dvojcku ni.
```

---

## 12. Kaj pomeni MW, Mvar, MVA, kV in kA

### MW

MW pomeni megawatt.

To je enota za aktivno moc:

```text
1 MW = 1,000,000 W
```

Aktivna moc je tista moc, ki opravlja koristno delo:

- poganja motorje,
- napaja industrijo,
- ogreva,
- osvetljuje,
- napaja elektronske naprave.

V datasetu so vrednosti porabe in proizvodnje zapisane predvsem v MW:

```text
load_*_p_mw
gen_*_p_mw
sgen_*_p_mw
```

`p_mw` pomeni aktivna moc `P` v MW.

### Mvar

Mvar pomeni megavolt-ampere reactive.

To je enota za jalovo moc `Q`. Jalova moc ne opravlja koristnega dela na enak nacin kot aktivna moc, je pa nujna za magnetna in elektricna polja v omrezju ter mocno vpliva na napetostni profil.

V digitalnem dvojcku imajo nekateri elementi stolpec:

```text
q_mvar
```

### MVA

MVA pomeni megavolt-ampere.

To je enota za navidezno moc `S`.

Povezava med aktivno, jalovo in navidezno mocjo je:

```text
S = P + jQ
|S| = sqrt(P^2 + Q^2)
```

Kjer je:

- `P`: aktivna moc,
- `Q`: jalova moc,
- `S`: navidezna moc.

### kV

kV pomeni kilovolt.

To je enota za elektricno napetost:

```text
1 kV = 1000 V
```

V tem modelu so busi na:

```text
380 kV
```

### kA

kA pomeni kiloampere.

To je enota za elektricni tok:

```text
1 kA = 1000 A
```

V tem modelu imajo linije:

```text
max_i_ka = 2.0 kA
```

To je tokovna meja, ki se uporablja pri oceni obremenitve linije.

---

## 13. Preprosta zveza med mocjo, napetostjo in tokom

Za enofazni sistem je osnovna intuicija:

```text
P = V * I
```

Kjer je:

- `P`: moc,
- `V`: napetost,
- `I`: tok.

Za trifazno AC omrezje je bolj uporabna poenostavljena formula:

```text
P ~= sqrt(3) * U * I * cos(phi)
```

Kjer je:

- `U`: medfazna napetost,
- `I`: tok,
- `cos(phi)`: faktor delovne moci.

Intuicija:

```text
vecja poraba ali proizvodnja
-> vecji pretoki moci
-> vecji tokovi po linijah
-> vecja obremenitev linij
```

Ce tok oziroma pretok po liniji preseze dovoljeno mejo, je linija preobremenjena.

---

## 14. Kaj pomeni loading percent

`loading_percent` pove, koliko je linija obremenjena glede na svojo dovoljeno mejo.

Poenostavljena formula:

```text
loading_percent = (dejanska obremenitev / dovoljena obremenitev) * 100
```

Ce je:

```text
loading_percent = 40
```

je linija priblizno 40 % obremenjena.

Ce je:

```text
loading_percent = 100
```

je linija na svoji meji.

Ce je:

```text
loading_percent > 100
```

je linija preobremenjena. V tem projektu je to eden glavnih razlogov za oznako `insecure`.

V koncnem datasetu imamo:

```text
max_line_loading_percent_basecase
max_line_loading_percent_contingency
```

Prvi stolpec opisuje najvecjo obremenitev linije v osnovnem primeru. Drugi opisuje najvecjo obremenitev v N-1 contingency primeru, vendar je v trenutni datoteki zapolnjen predvsem za primere, ki ostanejo `secure` po preverjanju.

---

## 15. Kaj je power flow

Power flow oziroma load flow je izracun stanja omrezja.

Vhod:

- topologija omrezja,
- impedanca linij,
- poraba po busih,
- proizvodnja po generatorjih,
- nastavitve napetosti,
- external grid/slack.

Izhod:

- napetost na vsakem busu,
- kot napetosti,
- pretok aktivne in jalove moci po linijah,
- obremenitev linij,
- izgube,
- informacija, ali je izracun konvergiral.

V notebooku se power flow izvede z:

```python
pp.runpp(net)
```

Ce power flow ne konvergira, to pomeni, da solver ni nasel stabilne resitve za dano stanje. V tem projektu se tak primer oznaci kot:

```text
insecure
```

---

## 16. Kaj je operating point

Operating point je posnetek stanja omrezja v eni uri.

V tem projektu operating point vsebuje:

- timestamp,
- 20 load vrednosti,
- 135 gen vrednosti,
- 110 sgen vrednosti,
- isto topologijo digitalnega dvojcka,
- rezultate power-flow simulacije,
- koncno `secure` / `insecure` labelo.

Primer:

```text
2023-01-01 08:00:00
load_0_p_mw = ...
gen_0_p_mw = ...
sgen_0_p_mw = ...
max_line_loading_percent_basecase = ...
status = secure/insecure
```

Mentalna slika:

```text
operating point = fotografija celotnega omrezja v eni uri
```

---

## 17. Kaj pomeni uniform

Datoteka:

```text
distributed_loads_uniform.csv
```

vsebuje load casovne vrste, porazdeljene po 20 load stolpcih:

```text
load_0, load_1, ..., load_19
```

Beseda `uniform` v tem kontekstu pomeni, da so obremenitve porazdeljene enakomerno oziroma po uniformnem pravilu med load elementi.

V prvih vrsticah datoteke imajo vsi `load_*` stolpci enako vrednost za isti timestamp. Primer:

```text
2023-01-01 00:00:00:
load_0 = load_1 = ... = load_19 = 219.0 MW
```

To pomeni, da je skupni load za ta timestamp priblizno:

```text
total_load = 20 * 219.0 MW = 4380 MW
```

Splosna formula:

```text
total_load(t) = sum(load_i(t))
```

Za trenutni `distributed_loads_uniform.csv`:

```text
stevilo vrstic = 8760
casovni razpon = 2023-01-01 00:00:00 do 2023-12-31 23:00:00
skupni load min = 0 MW
skupni load max = 10967 MW
skupni load average ~= 5464.99 MW
```

To je letni urni profil: 365 dni * 24 ur = 8760 ur.

---

## 18. Kaj vsebuje `distributed_generators.csv`

Datoteka:

```text
data/distributed_generators.csv
```

vsebuje casovne vrste proizvodnje.

Ima:

```text
8760 vrstic
casovni razpon = 2023-01-01 00:00:00 do 2023-12-31 23:00:00
```

Stolpci:

```text
135 gen stolpcev:  gen_0 ... gen_134
110 sgen stolpcev: sgen_0 ... sgen_109
1 timestamp stolpec
```

Skupaj:

```text
246 stolpcev
```

Ta datoteka pove, koliko aktivne moci v MW proizvaja vsak generator oziroma static generator v posamezni uri.

Pri pripravi dataseta notebook za vsak timestamp naredi:

```python
net.gen.at[i, "p_mw"] = gen_row[f"gen_{i}"]
net.sgen.at[i, "p_mw"] = gen_row[f"sgen_{i}"]
```

To pomeni, da se casovna vrsta proizvodnje neposredno vpise v digitalni dvojcek pred power-flow simulacijo.

---

## 19. Kaj vsebuje `simulation_security_labels_n-1.csv`

To je glavni supervised machine learning dataset.

Datoteka:

```text
data/simulation_security_labels_n-1.csv
```

ima:

```text
8769 vrstic
273 stolpcev
casovni razpon = 2023-01-01 00:00:00 do 2023-12-31 23:00:00
```

Opomba: osnovni letni urni profil ima 8760 timestampov, ta datoteka pa ima 8769 vrstic. Trenutna datoteka vsebuje 9 podvojenih timestampov na zacetku leta:

```text
2023-01-01 00:00:00 do 2023-01-01 08:00:00
```

Zato je pravilno dokumentirati trenutno datoteko kot 8769-vrsticno verzijo, ne pa predpostaviti, da ima natanko 8760 unikatnih operating pointov.

### Glavne skupine stolpcev

| Skupina | Stevilo | Primer |
|---|---:|---|
| Timestamp | 1 | `timestamp` |
| Label | 1 | `status` |
| Base-case security indicators | 3 | `max_line_loading_percent_basecase` |
| Contingency security indicators | 3 | `max_line_loading_percent_contingency` |
| Load features | 20 | `load_0_p_mw` |
| Gen features | 135 | `gen_0_p_mw` |
| Sgen features | 110 | `sgen_0_p_mw` |

### Label distribution

V trenutni datoteki:

| Label | Stevilo vrstic |
|---|---:|
| `secure` | 4497 |
| `insecure` | 4272 |

To je relativno uravnotezen dataset za binarno klasifikacijo.

### Security indikatorji

Stolpci:

```text
max_line_loading_percent_basecase
min_bus_voltage_pu_basecase
max_bus_voltage_pu_basecase
max_line_loading_percent_contingency
min_bus_voltage_pu_contingency
max_bus_voltage_pu_contingency
```

povedo, kaj se je zgodilo v simulaciji.

Za trenutni dataset:

```text
max_line_loading_percent_basecase:
  min ~= 5.49
  max ~= 175.33
  avg ~= 46.79

max_line_loading_percent_contingency:
  count = 4497
  min ~= 8.91
  max ~= 99.98
  avg ~= 45.20
```

Contingency stolpci so prazni pri 4272 vrsticah. To ustreza `insecure` primerom, kjer se simulacija ustavi oziroma se primer oznaci kot insecure, ko se najde krsitev ali non-convergence.

---

## 20. Kako nastane secure/insecure labela

Notebook:

```text
notebooks/prepare_security_classification_dataset.ipynb
```

ustvari dataset.

Postopek:

1. Nalozi digitalni dvojcek iz `digital_twin_ext_grid.json`.
2. Nalozi load casovne vrste iz `distributed_loads_uniform.csv`.
3. Nalozi generation casovne vrste iz `distributed_generators.csv`.
4. Za vsak timestamp vpise load, gen in sgen vrednosti v digitalni dvojcek.
5. Izvede base-case AC power flow.
6. Preveri base-case omejitve.
7. Ce je base case varen, izvede N-1 contingency simulacije.
8. Pri vsaki contingency simulaciji izklopi eno linijo ali en generator.
9. Ponovno izvede power flow.
10. Ce pride do preobremenitve, napetostne krsitve ali non-convergence, oznaci primer kot `insecure`.
11. Ce vse preverjene contingency simulacije ostanejo znotraj omejitev, oznaci primer kot `secure`.

### Base-case pravilo

Base case je osnovno stanje brez izpada elementa.

V notebooku je stanje `insecure`, ce velja vsaj eno od pravil:

```text
max_line_loading_base > 100
min_bus_voltage_base < 0.9
max_bus_voltage_base > 1.1
power flow does not converge
```

### N-1 pravilo

Ce base case nima krsitve, se preveri N-1.

Za vsako testirano linijo:

```text
izklopi eno linijo
-> run power flow
-> preveri line loading in bus voltage
```

Za vsak generator:

```text
izklopi en generator
-> run power flow
-> preveri line loading in bus voltage
```

Stanje je `insecure`, ce pri kateremkoli izpadu velja:

```text
max_line_loading_temp > 100
min_bus_voltage_temp < 0.9
max_bus_voltage_temp > 1.1
power flow does not converge
```

---

## 21. Kaj je N-1 kriterij

N-1 kriterij je osnovno varnostno pravilo v prenosnih elektroenergetskih omrezjih.

Pomen:

```text
Omrezje mora ostati varno tudi, ce odpove en sam element.
```

Ta element je lahko:

- ena linija,
- en generator,
- v splosnem tudi transformator,
- v drugih modelih lahko tudi drug pomemben element.

V tem projektu se po notebooku preverjajo:

```text
linije, razen line_45 Slack-to-Bus
generatorji gen
```

Ce omrezje po vseh testiranih posameznih izpadih ostane v dovoljenih mejah, je operating point:

```text
secure
```

Ce ze en sam izpad povzroci problem, je operating point:

```text
insecure
```

Formalno lahko napisemo:

```text
secure(x) = true,
ce za base case in vse testirane contingency c velja:

loading_max(x, c) <= 100 %
0.9 <= V_min(x, c)
V_max(x, c) <= 1.1
power_flow_converged(x, c) = true
```

Ce katerikoli pogoj odpove:

```text
secure(x) = false
```

oziroma:

```text
status = insecure
```

---

## 22. Kaj pomeni insecure

`insecure` ne pomeni nujno blackout.

V tem projektu `insecure` pomeni:

```text
operating point ne izpolnjuje simulacijskega steady-state N-1 security kriterija
```

Razlogi:

- base-case power flow ne konvergira,
- base-case linija preseze 100 % loading,
- base-case bus napetost pade pod 0.9 pu,
- base-case bus napetost preseze 1.1 pu,
- N-1 contingency povzroci katerokoli od zgornjih krsitev,
- N-1 contingency power flow ne konvergira.

Zato je `insecure` tehnicna oznaka za krsitev kriterija, ne neposredna napoved popolnega izpada sistema.

---

## 23. Kako je grid povezan

Grid je graf.

V grafovski mentalni sliki:

```text
bus = vozlisce grafa
line = povezava med dvema vozliscema
load = poraba prikljucena na vozlisce
gen/sgen = proizvodnja prikljucena na vozlisce
ext_grid = referencna povezava z zunanjim sistemom
```

Primer:

```text
1208 --- line 10950 --- 1209
1208 --- line 3601  --- 1210
1208 --- line 10949 --- 1210
```

To pomeni, da lahko med istima busoma obstaja vec razlicnih linij. V zgornjem primeru sta med `1208` in `1210` dve liniji, kar poveca redundanco.

Redundanca je pomembna za N-1:

```text
ce ena linija odpove,
se mora moc preusmeriti po preostalih povezavah
```

Ce preostale povezave prenesejo pretoke brez preobremenitev in napetostnih problemov, je stanje lahko secure.

Ce se pretok preusmeri tako, da druga linija preseze 100 %, je stanje insecure.

---

## 24. Zakaj visja poraba lahko poveca tveganje

Vecja poraba obicajno pomeni vecje pretoke moci po omrezju.

Preprosta veriga:

```text
vecji load
-> generatorji morajo proizvesti vec
-> moc tece po linijah od proizvodnje do porabe
-> tokovi po linijah narastejo
-> line loading naraste
-> N-1 izpad lahko hitreje povzroci preobremenitev
```

Vendar povezava ni vedno 1:1.

Dve uri imata lahko enak total load, vendar drugacno varnost:

- poraba je lahko drugace prostorsko razporejena,
- proizvodnja je lahko drugace razporejena,
- kriticne linije so lahko razlicno obremenjene,
- razlicni generatorji lahko delujejo z razlicnimi mocmi,
- N-1 izpad lahko prizadene razlicne dele omrezja.

Zato security classification ni samo napoved porabe. Je ocena varnosti celotnega operating pointa.

---

## 25. Kaj dela Random Forest model

Notebook:

```text
notebooks/train_random_forest_security_classifier.ipynb
```

trenira binarni klasifikator.

Target:

```python
secure   -> 1
insecure -> 0
```

Model:

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

Train/test split:

```python
test_size = 0.2
stratify = target
random_state = 42
```

Pomembna podrobnost: notebook iz featurejev odstrani simulacijske indikatorje:

```text
timestamp
status
status_binary
max_line_loading_percent_basecase
min_bus_voltage_pu_basecase
max_bus_voltage_pu_basecase
max_line_loading_percent_contingency
min_bus_voltage_pu_contingency
max_bus_voltage_pu_contingency
```

To pomeni, da se model uci predvsem iz vhodnih operativnih vrednosti:

```text
load_*_p_mw
gen_*_p_mw
sgen_*_p_mw
```

Model se shrani kot:

```text
models/random_forest_model.pkl
```

Po zadnjih lokalnih metric datotekah je ena od shranjenih verzij dosegla:

```text
accuracy ~= 0.933
```

To je rezultat na testnem delu podatkov iz notebooka, ne fizikalna lastnost omrezja.

---

## 26. Kaj pomeni `p_insecure`

`p_insecure` je verjetnost, ki jo lahko vrne klasifikacijski model.

Primer:

```text
p_insecure = 0.06
```

pomeni:

```text
model ocenjuje 6 % verjetnost, da je operating point insecure
```

To ni direktna meritev iz omrezja. To je modelna ocena.

Pri binarni klasifikaciji je model najbolj negotov blizu:

```text
p_insecure ~= 0.5
```

Ce je:

```text
p_insecure ~= 0.01
```

je model precej preprican, da je stanje secure.

Ce je:

```text
p_insecure ~= 0.99
```

je model precej preprican, da je stanje insecure.

Ce je:

```text
p_insecure ~= 0.5
```

je model neodlocen. Tak primer je posebej zanimiv za Active Learning.

---

## 27. Kaj je Active Learning v tem projektu

N-1 simulacije so drage, ker je treba za vsak operating point izvesti veliko power-flow izracunov:

```text
base case
+ izpad linije 1
+ izpad linije 2
+ ...
+ izpad generatorja 1
+ izpad generatorja 2
+ ...
```

Active Learning poskusa zmanjsati stevilo potrebnih simulacij.

Namesto:

```text
simuliraj vse mozne primere
```

uporabimo:

```text
model napove varnost
-> poiscemo negotove ali informativne primere
-> samo te posljemo v digitalni dvojcek
-> dobimo ground-truth labelo
-> dodamo v dataset
-> retrain model
```

V tem projektu je pomembna interpretacija:

- operater ne rocno labelira secure/insecure,
- operater lahko izbira ali potrjuje zanimive primere,
- ground truth labelo proizvede Digital Twin z N-1 simulacijo.

---

## 28. Tipicne Active Learning strategije

### Uncertainty sampling

Izberemo primere, kjer je model najbolj negotov.

Za binarni problem:

```text
uncertainty je najvecja pri p ~= 0.5
```

Poenostavljena mera:

```text
uncertainty = 1 - max(p_secure, p_insecure)
```

Vecja vrednost pomeni vecjo negotovost.

### Margin sampling

Margin meri razliko med prvima dvema najverjetnejsima razredoma.

Pri binarni klasifikaciji:

```text
margin = |p_secure - p_insecure|
```

Manjsi margin pomeni vecjo negotovost.

Primer:

```text
p_secure = 0.51
p_insecure = 0.49
margin = 0.02
```

To je zelo negotov primer.

### Entropy sampling

Entropy meri neurejenost oziroma negotovost porazdelitve verjetnosti.

Za dva razreda:

```text
H(p) = - p_secure * log(p_secure) - p_insecure * log(p_insecure)
```

Entropy je najvecja, ko sta razreda skoraj enako verjetna.

### Random sampling

Random sampling izbere primere nakljucno.

To je pomemben baseline:

```text
Ce Active Learning ne premaga random izbire,
potem strategija morda ne dodaja vrednosti.
```

---

## 29. Kratek opis vsake projektne datoteke

### `data/digital_twin_ext_grid.json`

Pandapower digitalni dvojcek omrezja.

Vsebuje:

- 35 busov,
- 46 linij,
- 20 load elementov,
- 135 generatorjev,
- 110 static generatorjev,
- 1 external grid,
- tehnicne parametre za power-flow simulacije.

To je osnovni staticni model omrezja.

### `data/distributed_loads_uniform.csv`

Letni urni load profil.

Vsebuje:

- 8760 vrstic,
- 20 load stolpcev,
- timestamp od 2023-01-01 00:00:00 do 2023-12-31 23:00:00.

Uporablja se za nastavitev porabe po load elementih.

### `data/distributed_generators.csv`

Letni urni generation profil.

Vsebuje:

- 8760 vrstic,
- 135 `gen_*` stolpcev,
- 110 `sgen_*` stolpcev,
- timestamp.

Uporablja se za nastavitev proizvodnje v digitalnem dvojcku.

### `data/simulation_security_labels_n-1.csv`

Koncen labeled dataset za machine learning.

Vsebuje:

- timestamp,
- `status` labelo,
- base-case simulacijske indikatorje,
- contingency simulacijske indikatorje,
- load/gen/sgen featureje.

To je dataset, iz katerega se trenira security classifier.

### `notebooks/prepare_security_classification_dataset.ipynb`

Notebook za generiranje labeled dataseta.

Izvede:

- branje digitalnega dvojcka,
- branje load/generation casovnih vrst,
- vpis casovnih vrednosti v omrezje,
- base-case power flow,
- N-1 contingency simulacije,
- zapis `secure` / `insecure` oznake,
- izvoz `simulation_security_labels_n-1.csv`.

### `notebooks/train_random_forest_security_classifier.ipynb`

Notebook za treniranje ML modela.

Izvede:

- branje labeled dataseta,
- mapiranje `secure -> 1`, `insecure -> 0`,
- odstranitev neprimernih ali simulacijskih stolpcev iz featurejev,
- train/test split,
- treniranje Random Forest klasifikatorja,
- izracun metrik,
- shranjevanje modela.

### `models/random_forest_model.pkl`

Shranjeni Random Forest model.

Uporablja se za hitro napovedovanje:

```text
nov operating point -> p_secure / p_insecure -> predicted status
```

Model je priblizek digitalnega dvojcka. Digitalni dvojcek je se vedno vir ground-truth label, kadar izvedemo N-1 simulacijo.

---

## 30. Mentalna slika celotnega sistema

Najbolj uporabna ucna slika:

```text
digital_twin_ext_grid.json
  = staticna mreza: busi, linije, generatorji, loads

distributed_loads_uniform.csv
  = kako se poraba spreminja po urah

distributed_generators.csv
  = kako se proizvodnja spreminja po urah

prepare_security_classification_dataset.ipynb
  = za vsako uro vstavi porabo/proizvodnjo v mrezo
  = izvede power flow in N-1 simulacije
  = ustvari labelo secure/insecure

simulation_security_labels_n-1.csv
  = ucni dataset za klasifikacijo

train_random_forest_security_classifier.ipynb
  = trenira model

random_forest_model.pkl
  = hiter prediktor varnosti za nove operating pointe
```

---

## 31. Kaj si mora bralec najbolj zapomniti

1. Bus je vozlisce omrezja.
2. Line povezuje dva busa.
3. Load porablja moc.
4. Gen in sgen proizvajata moc.
5. External grid oziroma slack bus je referenca za power-flow izracun.
6. V tem digitalnem dvojcku so vsi busi na 380 kV.
7. Eksplicitnih transformatorjev v trenutni datoteki ni.
8. MW meri aktivno moc.
9. kV meri napetost.
10. kA meri tok.
11. `pu` pomeni relativno vrednost glede na nazivno vrednost.
12. Power flow izracuna napetosti in pretoke po omrezju.
13. N-1 pomeni: omrezje mora preziveti izpad enega elementa.
14. `secure` pomeni, da base case in testirane N-1 situacije ostanejo znotraj omejitev.
15. `insecure` pomeni krsitev omejitev ali non-convergence.
16. Random Forest je hiter ML priblizek simulacijskemu procesu.
17. Active Learning izbira najbolj informativne primere za dodatno simulacijo.

---

## 32. Kratek povzetek v enem odstavku

Smart Energy dataset je digital-twin-based N-1 security classification dataset za elektroenergetsko prenosno omrezje. Staticni model omrezja je zapisan v `digital_twin_ext_grid.json` in vsebuje 35 busov, 46 linij, 20 load elementov, 135 generatorjev, 110 static generatorjev in 1 external grid na 380 kV nivoju. Letne urne casovne vrste porabe in proizvodnje se vstavijo v digitalni dvojcek, nato se za vsak operating point izvede base-case power flow in N-1 preverjanje izpadov linij ter generatorjev. Ce pride do preobremenitve linije, napetostne krsitve ali non-convergence, se stanje oznaci kot `insecure`; sicer kot `secure`. Koncni CSV se uporablja za treniranje Random Forest klasifikatorja, ki hitro napove varnost novega operating pointa, medtem ko digitalni dvojcek ostaja vir ground-truth label pri dodatnih simulacijah.
