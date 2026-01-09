# SteatosisU-UNet

README in italiano basato sul PDF fornito (bozza). Se il PDF contiene informazioni specifiche (dataset, iperparametri, risultati numerici, riferimenti bibliografici), per favore forniscilo o indicami dove è salvato nel repository così posso aggiornare i dettagli.

## Descrizione

SteatosisU-UNet è un progetto per la segmentazione della steatosi epatica (o tessuti con steatosi) utilizzando una variante di U-Net ("U-UNet" / UNet-like). Lo scopo è identificare e quantificare le aree di steatosi nelle immagini (es. istologiche o ecografiche) per aiutare nell'analisi automatizzata.

Questa README è una bozza creata senza accesso diretto al PDF originale: ho inserito le informazioni tipiche e alcuni placeholder dove il PDF probabilmente fornisce i valori esatti (es. dimensioni dataset, metriche, iperparametri, risultati sperimentali).

## Contenuto principale (atteso)

- Modellistica: architettura U-Net modificata (descrivere variazioni: profondità, blocchi, attention, se presenti).
- Dataset: descrizione del dataset usato (numero immagini, tipo di immagini, risoluzione, annotazioni/maschere), split train/val/test.
- Preprocessing: normalizzazione, resize, data augmentation.
- Training: loss function (es. DiceLoss, BCE + Dice), ottimizzatore, learning rate, numero di epoche, batch size.
- Valutazione: metriche usate (Dice, IoU, Precision, Recall), procedure di convalida.
- Risultati: riassunto numerico dei risultati sperimentali e confronto con baseline (placeholder in questa bozza).

## Requisiti

- Python 3.8+ (consigliato 3.8–3.11)
- GPU con CUDA (opzionale ma raccomandata per training)
- Dipendenze tipiche: torch, torchvision, numpy, scikit-image, opencv-python, tqdm, matplotlib

Esempio di ambiente (se non esiste un file `requirements.txt` nel repo):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy scikit-image opencv-python tqdm matplotlib
# Aggiungere altre dipendenze specifiche del progetto se presenti
```

Se nel repository è presente un `requirements.txt` o `environment.yml`, usare quello al posto della lista sopra.

## Installazione e setup

1. Clona il repository

```bash
git clone <url-del-repo>
cd SteatosisU-UNet/SteatosisU-UNet
```

2. Crea e attiva un ambiente virtuale (vedi sezione Requisiti)

3. Installa le dipendenze

```bash
pip install -r requirements.txt  # se presente
```

## Struttura dati attesa

La seguente è una struttura tipica richiesta dagli script di training/inference:

```
dataset/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

- Le immagini possono essere in formato PNG, TIFF o JPEG.
- Le maschere sono immagini binarie o a valori interi con la stessa risoluzione delle immagini di input.

Adattare gli script se il dataset ha una struttura diversa.

## Uso (esempi generici)

Ecco comandi d'esempio che potresti adattare agli script presenti nel repo.

Train (esempio generico):

```bash
python train.py \
  --data-dir dataset \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --model-path outputs/unet_best.pth
```

Inferenza singola immagine (esempio generico):

```bash
python predict.py \
  --model outputs/unet_best.pth \
  --input path/to/image.png \
  --output path/to/pred_mask.png
```

Valutazione su test-set (esempio generico):

```bash
python evaluate.py --data-dir dataset/test --model outputs/unet_best.pth
```

## Iperparametri consigliati (placeholder)

- Ottimizzatore: Adam, lr = 1e-4
- Loss: BCE + DiceLoss (o solo DiceLoss)
- Batch size: 4–16 (dipende dalla GPU)
- Epoche: 50–200

Questi valori sono generici: sostituire con quelli riportati nel PDF per replicare esattamente gli esperimenti.

## Metriche di valutazione

- Dice Coefficient (F1)
- Intersection over Union (IoU)
- Precision / Recall
- Pixel Accuracy

Visualizzazioni utili: curve di training (loss/metriche), matrici di confusione su pixel-level o region-level, esempi qualitativi (immagine, maschera ground-truth, predizione).

## Risultati (placeholder)

Nel PDF sono riportati risultati quantitativi — inserire qui i valori esatti (es. Dice medio, IoU, std) e le figure principali (tabelle e grafici). Se vuoi, posso aggiornare questa sezione dopo che mi fornisci i numeri dal PDF.

## Riproducibilità

- Impostare seed globale per CPU e GPU
- Salvare la configurazione degli esperimenti (file config o YAML)
- Registrare il commit git e l'ambiente (pip freeze > requirements_used.txt)

## Dove migliorare / possibili estensioni

- Aggiungere meccanismi di regularizzazione (dropout, weight decay)
- Usare tecniche di ensembling o test-time augmentation
- Provare architetture alternative (Attention U-Net, UNet++ ecc.)
- Validazione cross-validation per dataset piccoli

## Citazione e contatti

Se usi questo lavoro, aggiungi la citazione corretta tratta dal PDF. Se non disponi della citazione, invia il PDF e la aggiungerò.

Per domande o contributi: apri un issue nel repository o contatta l'autore principale (inserire qui il nome/email se presente nel PDF).

## Assunzioni e note

1. Non ho accesso al PDF nel contesto corrente: ho creato la README come bozza basata su pratiche comuni per progetti di segmentazione con U-Net.
2. Ho inserito placeholder per i dettagli che probabilmente sono nel PDF: dataset (dimensioni, provenienza), iperparametri esatti, risultati numerici e riferimenti bibliografici.
3. Se vuoi, posso aggiornare immediatamente il README con i dettagli esatti se carichi il PDF o incolli qui le sezioni rilevanti.

---

Se desideri, procedo ora a:

1. Cercare nel repository file che fanno training/inferenza (es. `train.py`, `predict.py`) e integrare istruzioni precise alla sezione "Uso".
2. Aggiornare i comandi di installazione con il reale `requirements.txt` se presente.
3. Inserire i risultati e la citazione dal PDF non appena me lo fornisci.

Indicami come procedere: vuoi che cerchi i file di script nel repository e aggiorni la README con istruzioni precise (opzione consigliata)?
