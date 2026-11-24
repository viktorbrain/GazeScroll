# GazeScroll – modellváz

Mobilos, előlapi kamerás tekintetbecslés + olvasás alapú automatikus görgetés.
A terv: „Mobilos eye tracker – modellterv” (Claude Docs).

```
gazescroll/
  features.py   MediaPipe 478 pont -> 14 elemű feature-vektor (írisz u/v, EAR, fejtartás, arc helyzete)
  model.py      lineáris ridge (skála-padlóval) + RLS online újrakalibrálás
  filters.py    One Euro filter
  reading.py    olvasásdetektor (előre-arány + sorvégi visszaugrás) és görgetési logika
  adapt.py      tanuló görgetés: kézi korrekciókból és az olvasási tempóból hangolja a paramétereket
  synth.py      szintetikus szem/fej modell + zárt hurkú olvasásszimuláció
tests/          pytest (27 teszt)
evaluate.py     szintetikus benchmark, Markdown táblát ír ki
replay.py       a telefonos tesztoldal naplójának visszajátszása a Python detektoron
web/index.html  telefonos tesztoldal (MediaPipe Tasks Vision, ugyanaz a logika JS-ben)
index.html      ugyanaz a gyökérben, a GitHub Pages-hez
```

## Futtatás

```bash
pip install numpy pytest
python -m pytest -q
python evaluate.py
python replay.py gazescroll-log-*.json --scroll-zone 0.6
```

## Telefonos teszt

A kamera csak HTTPS-en érhető el. Közvetlen link (GitHub Pages bekapcsolása után): https://viktorbrain.github.io/gazescroll/

Más lehetőségek:

- vagy pl. `viktornagy.hu/gazescroll/index.html`.
- helyi gépen: `python -m http.server` + `adb reverse tcp:8000 tcp:8000`, majd a telefonon `http://localhost:8000` (a localhost biztonságos környezetnek számít).

A gombok: felbontás (640×480 / 1280×720), kalibráció (9 pont × 2 kör + 4 pontos validáció), tekintetpont ki/be, auto-görgetés ki/be, napló mentése (JSON), tanult profil törlése.

**Tanuló görgetés:** ha egy automatikus görgetés után 3 s-on belül visszahúzod a szöveget, az „korai” címke (a görgetési zóna lejjebb kerül); ha olvasás közben magad görgetsz lefelé, az „késő” címke (a zóna feljebb kerül). A sorvégek közti idő az olvasási tempó. A profil a böngészőben marad (localStorage), és a napló is tartalmazza.

**Napló:** képkockánként tekintet, feature-ök, állapot, görgetési pozíció, írisz átmérője pixelben, arc szélessége, fényerő (0–255); plusz validációs hiba és a kézi korrekciók.
A „Demó kamera nélkül” mód az érintést használja tekintetként – a detektor logikáját így kamera nélkül is ki lehet próbálni.

## Keskeny szemrés / gyenge kalibráció

- **Adaptív pislogásszűrés:** a pislogást a saját szemnyitottságodhoz mérjük (a megszokott érték 60%-a alatt), nem fix 0,18-as EAR-küszöbhöz. Ez utóbbi keskeny szemrésnél szinte minden képkockát eldobott.
- **Hosszabb pontidő:** ha egy kalibrációs ponton kevés képkockán látszik a szem, a pont legfeljebb 2,6 s-ig kint marad. Lent élőben látod, hány százalékban látszik a szem.
- **Egyszerű mód (csak vízszintes, 3 pont):** csak a bal–közép–jobb irányt tanulja meg. Minden sorvégi visszaugrásnál egy sort görget, így az éppen olvasott sor egy magasságban marad. Függőleges becslés nem kell hozzá.
- **Automatikus visszalépés:** ha a 18 pontból 6-nál kevesebb sikerül, vagy a függőleges hiba nagy, magától erre a módra vált.
- A MediaPipe arckeresési küszöbei 0,5-ről 0,3-ra csökkentek.

## v4 – első valós napló alapján

- **Pislogás csak rövid szemhéjzárás** (≤ 0,4 s). A hosszabb alacsony EAR lefelé nézés, nem pislogás: ezért dobta el korábban az alsó kalibrációs pontokat.
- **Kalibrációs jelentés:** pontonként arc-, szem- és mintaszám, és a felső / alsó sor láthatósága. Bekerül a naplóba, és sikertelen kalibrációnál ki is írja.
- **Indítás kalibráció nélkül (kényszerített):** önhangoló vízszintes móddal indul.
- **Önhangoló x (RollingNorm):** az utolsó 12 s nyers x-ének 10/90. percentilise lesz a sor eleje és vége, így a fejfordulás és a telefon elmozdulása nem viszi el.
- **Tempótartás:** vízszintes módban, ha olvasol, de 1,3 sornyi ideig nem jött görgetés, magától görget egy sort (a tempó tanult).
- **Biccentés:** gyors fejbiccentés (a dőlésszög ugrása > 0,08, és 0,8 s-on belül visszatér) a képernyő 1/3-ával görget. Kikapcsolható.
- **Ismert korlát:** vízszintes módban a véletlenszerű pásztázás néha sorvégnek látszik (szimulációban 11–20 téves sor/perc pásztázás közben).
