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
tests/          pytest (20 teszt)
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
