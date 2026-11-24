# GazeScroll – kéréslista

Viktor kérései és állapotuk. Új kérés a lista végére kerül.

| # | Kérés | Állapot | Verzió |
| --- | --- | --- | --- |
| 1 | Modellváz és terv, utána app; tesztelhető legyen | kész (Python váz + tesztek + telefonos tesztoldal) | v1 |
| 2 | Verziókövetés a GazeScroll repóban, GitHub Pages | kész | v1 |
| 3 | Tanuló görgetés (kézi korrekciókból) | kész | v2 |
| 4 | Arc/szem kivágás nagyobb felbontásból (zoom) | részben: választható 1280×720; saját szemkivágásos modell nyitott | v2 |
| 5 | Invertált fekete-fehér kép a pupillához | elvetve (a MediaPipe színes képre tanult); CLAHE csak klasszikus pupilla-kereséshez | – |
| 6 | Keskeny szemrés kompenzálása, továbblépés gyengébb kalibrációval | kész: adaptív pislogás, egyszerű (vízszintes) mód, automatikus visszalépés | v3 |
| 7 | Fejbiccentés mint vezérlés | kész: biccentés = 1/3 képernyő görgetés, kapcsolható | v4 |
| 8 | A görgetés ne maradjon le az alsó soroknál | kész: tempótartás, önhangoló x | v4 |
| 9 | Normál kalibráció hibát ír 9x%-os láthatóság mellett | javítva: lefelé nézés ≠ pislogás; részletes kalibrációs jelentés | v4 |
| 10 | Kényszerített indítás kalibráció nélkül | kész: „Indítás kalibráció nélkül” gomb | v4 |
| 11 | Kamerakép jobb felülre, kisebb vagy vágott kép | kész: kis szemkivágás jobb felül; koppintás: szem / arc / elrejtés | v5 |

## Nyitott ötletek

- Súgógép-szerű tempómód: a kamera csak a jelenlétet figyeli, biccentéssel gyorsít/lassít.
- Saját szemkivágásos (appearance-based) modell a nagyobb felbontású képből.
- Biccentés-érzékenység beállítása, irány szerinti parancsok (le = tovább, fel = vissza).
- Laboros validálás a SZESIM eye trackerrel.
- Android app (CameraX + MediaPipe, AccessibilityService).
