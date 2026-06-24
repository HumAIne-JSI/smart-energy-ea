# ⚡ Day-Ahead Load Forecast in N-1 Security Classification

> ⚠️ This document is written in Slovenian. See [smart-energy-data-description.md](../smart-energy-data-description.md) sections 13–15 for the equivalent English explanation.

Ta dokument na kratko razloži, kaj prikazujeta spodnja grafa, kako sta povezana in zakaj ju ne smemo razumeti kot isto stvar. Prvi graf opisuje **napoved obremenitve omrežja**, drugi pa **oceno varnostnega tveganja** za iste urne operating pointe.

---

## 1. 📈 Day-Ahead Total Load Forecast

![Day-Ahead Total Load Forecast](../figures/load_forecast.png)

Prvi graf prikazuje **napoved skupne električne porabe oziroma obremenitve omrežja** za posamezne ure dneva.

- **X-os:** ura dneva
- **Y-os:** skupna obremenitev omrežja v MW
- **Pomen:** koliko bo omrežje predvidoma obremenjeno v posamezni uri

Primer interpretacije:

- okoli 08:00 je napovedana poraba približno 7200 MW
- okoli 09:00 je napovedana poraba približno 7300 MW
- zvečer se obremenitev zmanjša, npr. okoli 21:00 na približno 5400 MW

Ta graf torej odgovarja na vprasanje:

> **Koliko električne porabe pričakujemo v vsaki uri dneva?**

---

## 2. 🛡️ N-1 Day-Ahead Security Classification

![N-1 Day-Ahead Security Classification](../figures/security_classification.png)

Drugi graf prikazuje **verjetnost, da je stanje omrežja v posamezni uri insecure**, če ga ocenjujemo po N-1 kriteriju.

- **X-os:** ura dneva
- **Y-os:** verjetnost insecure stanja (`p_insecure`)
- **Pomen:** kako tvegano je stanje celotnega omrežja pri danem operating pointu

Primer interpretacije:

- okoli 08:00 je `p_insecure` približno 0.06
- okoli 09:00 je `p_insecure` približno 0.04
- okoli 13:00 je `p_insecure` približno 0.02

To ni napoved porabe. To je **ocena varnosti omrežja**.

Ta graf odgovarja na vprasanje:

> **Ali bo omrežje pri tej obremenitvi in danem scenariju verjetno varno tudi ob N-1 izpadu?**

---

## 3. 🔗 Kako sta grafa povezana?

Najbolj preprosto:

> **Load forecast je vhod, security classification pa rezultat oziroma ocena tveganja.**

Približen tok podatkov je:

```text
Day-ahead load forecast
-> operating point za vsako uro
-> N-1 security classification model / analiza
-> probability of insecurity
-> security classification graf
```

Prvi graf pomaga pripraviti urne operating pointe. Drugi graf nato za te operating pointe oceni, ali je omrežje varno oziroma kako verjetno je insecure stanje.

---

## 4. ⚠️ Zakaj višja poraba lahko pomeni višje tveganje?

Ko je poraba visoka, je omrežje praviloma bolj obremenjeno:

```text
višja poraba
-> večji pretoki po linijah
-> večja možnost preobremenitev
-> večja možnost insecure stanja
```

Zato lahko pogosto pricakujemo:

> **višji load -> višji `p_insecure`**

V prikazanem primeru se to delno vidi. Zjutraj se load močno poveča, predvsem med 08:00 in 12:00, ko je obremenitev okoli 7000-7300 MW. V istem obdobju je tudi `p_insecure` nekoliko višji, približno med 0.04 in 0.06.

Pomembno pa je, da so te vrednosti še vedno nizke. Na primer `p_insecure = 0.06` pomeni, da model vidi rahlo povečano tveganje, ne pa kritičnega stanja.

---

## 5. 🧩 Zakaj povezava ni 1:1?

Drugi graf ni samo kopija prvega grafa. Security classification ni odvisen samo od skupne porabe, ampak tudi od drugih lastnosti operating pointa:

- kje v omrežju je poraba
- kje je proizvodnja
- kateri generatorji delujejo
- kakšni so pretoki po posameznih linijah
- kakšne so napetosti po busih
- kakšna je topologija omrežja
- kateri element odpove v N-1 analizi

Zato sta lahko dve uri z zelo podobnim total loadom varnostno različni. Ena ura ima lahko visoko skupno porabo, vendar dobro razporejene pretoke. Druga ura ima lahko nižjo skupno porabo, vendar je kritična linija že zelo obremenjena.

---

## 6. 🕒 Kaj pomeni "day-ahead"?

**Day-ahead** ni element omrežja. Pomeni **časovni horizont**.

V energetiki se izraz day-ahead uporablja za napovedi ali analize, ki se pripravijo za naslednji operativni dan oziroma za 24-urni profil dneva.

V tem primeru:

- **day-ahead load forecast** pomeni napoved skupne porabe po urah
- **day-ahead security classification** pomeni oceno varnosti po urah
- vsak stolpec predstavlja en urni **operating point**

Operating point pomeni snapshot stanja celotnega omrežja v določeni uri:

- čas
- poraba po vozliščih
- proizvodnja
- topologija
- pretoki po linijah
- napetostni pogoji

---

## 7. 🔌 Kaj pomeni N-1 kriterij?

N-1 analiza preverja, kaj se zgodi, če odpove en element omrežja, na primer:

- linija
- generator
- transformator

Pomembno: drugi graf ne prikazuje ene same linije ali enega samega elementa. Vsak stolpec prikazuje oceno za **celotno omrežje v eni uri**.

Na primer stolpec za 08:00 pomeni:

```text
operating point omrežja ob 08:00
-> preverjanje oziroma modeliranje N-1 varnosti
-> p_insecure = 0.06
```

To pomeni, da je stanje po modelu nekoliko bolj tvegano kot pri drugih urah, vendar še vedno verjetno secure.

---

## 8. 🎯 Povezava z Active Learning

Za Active Learning je posebej pomemben drugi graf, ker lahko predstavlja **candidate pool**: množico ur oziroma operating pointov, ki jih lahko sistem predlaga za dodatno preverjanje.

Vendar najbolj zanimive ure niso nujno tiste z najvišjim `p_insecure`. Pri binarni klasifikaciji je model najbolj negotov takrat, ko je verjetnost blizu:

```text
p_insecure ~= 0.5
```

Na prikazanem grafu so vrednosti precej nizke, približno med 0.01 in 0.06. To pomeni, da model za ta dan večinoma vidi secure stanje in ni zelo negotov.

Retraining gumb je zato vezan na **konkreten security classification instance**, ne neposredno na load forecast graf. Ko operator izbere uro in sproži retraining, se tipično izbrani operating point dodatno obdela, rezultat se doda v učno množico, nato pa backend sproži ponovno učenje modela.

---

## 9. 🌉 Najkrajša mentalna slika

> **Prvi graf = koliko teže damo na most.**  
> **Drugi graf = kakšna je verjetnost, da most pri tej teži postane nevaren.**

Večja teža pogosto pomeni večje tveganje, vendar ni edini dejavnik. Pomembno je tudi, kako je "most" zgrajen, kje je obremenitev, kateri del odpove in kakšno je trenutno stanje sistema.

V kontekstu elektroenergetskega omrezja:

- **load forecast** = napoved obremenitve omrežja
- **security classification** = ocena, ali omrežje zdrži to obremenitev tudi ob N-1 izpadu

---

## 10. ✅ Ključni povzetek

- Prvi graf prikazuje **napoved porabe** po urah.
- Drugi graf prikazuje **verjetnost insecure stanja** po urah.
- Grafa sta povezana, ker load forecast pomaga tvoriti operating pointe za security classification.
- Vseeno drugi graf ni samo preslikava prvega, ker je varnost odvisna od celotnega stanja omrežja.
- V prikazanem dnevu je zjutraj load visok, zato se nekoliko poveča tudi `p_insecure`.
- Vrednosti `p_insecure` so kljub temu nizke, zato model vecinoma ocenjuje stanje kot secure.
- Za Active Learning so zanimivi predvsem operating pointi, kjer je model negotov, ne nujno samo tisti z najvišjo napovedano porabo.
