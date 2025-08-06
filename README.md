# Titanic Survival Prediction App

Una moderna applicazione Streamlit per l'analisi e la predizione della sopravvivenza sul Titanic.

## Quick Start

1. Attiva il virtual environment:
   ```bash
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate   # Linux/Mac
   ```

2. Installa le dipendenze:

   Ecco come generare il file requirements.txt se non è stato fatto precedentemente
   ```bash
   pip freeze > requirements.txt
   ```
   Per istallare tutte le dipendenze per il progetto:
   ```bash
   pip install -r requirements.txt
   ```

3. Esegui l'applicazione:
   ```bash
   streamlit run app.py
   ```

## Struttura del Progetto

```
titanic-streamlit-dashboard/
├── app.py                      # File principale dell'app
├── src/                        # Codice sorgente
│   ├── components/             # Componenti riusabili
│   ├── data/                   # Dati del progetto
│   ├── models/                 # Modelli ML
│   ├── utils/                  # Utilities
│   └── config.py              # Configurazioni
├── pages/                      # Pagine multiple
├── assets/                     # Risorse statiche
├── tests/                      # Test unitari
├── docs/                       # Documentazione
└── .streamlit/                 # Configurazione Streamlit
```

### Analisi strutturale del progetto
Per osservare la struttura del progetto senza l'ambiente virtuale, eseguiamo il commanndo nel terminale con powershell:
```powershell
   Get-ChildItem -Recurse | Where-Object { $_.FullName -notmatch "venv" } | Select-Object FullName
```
## Tecnologie Utilizzate

- **Streamlit**: Framework per l'app web
- **Pandas**: Manipolazione dati
- **Scikit-learn**: Machine Learning
- **Matplotlib/Seaborn**: Visualizzazioni
- **Plotly**: Grafici interattivi

## Funzionalità

- Analisi esplorativa dei dati
- Visualizzazioni interattive
- Modelli di machine learning
- Predizioni in tempo reale
- Dashboard con metriche
- Interfaccia utente moderna

## Sviluppo

Il progetto è stato migrato da un Jupyter Notebook (`docs/data_analysis.ipynb`) a un'applicazione Streamlit modulare e scalabile. il file Jupyter Notebook (`data_analysis.ipynb`) è ancora in fase di completamento. ma la versione streamlit è già operativa.
