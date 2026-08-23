# EEG-BACKEND

A small Flask service that takes a couple of seconds of raw EEG from a consumer
meditation headband and returns a reading of what the wearer's attention was
doing. It is the analysis half of a two-repo project; the browser front end
lives in [EEG-UI](https://github.com/Nishchay-Bhudia/EEG-UI) and does the
Bluetooth work.

The vocabulary it reports in is deliberately not clinical. States are labelled
with terms from Patanjali's Yoga Sutras (the five Chitta Bhumis, from Mudha
through to Niruddha), plus Swara Nadi lateralisation and the three Gunas. That
was the point of the project: map measurable EEG features onto the framework a
meditation teacher already uses, rather than handing them a spectrogram.

## What it does

Three endpoints, no database, no state between requests.

- `GET /status` returns whether the classifier is up.
- `POST /analyze` takes raw microvolt samples, `{ "eeg_data": [[...] x 4], "sample_rate": 256 }`,
  and does the whole pipeline.
- `POST /analyze/bands` takes band powers that the client already computed, for
  when the browser has fallen back to its own FFT and just wants labels.

A successful response carries the Chitta Bhumi and its probability spread,
contemplative depth, Swara, Guna percentages, relative power in six bands,
frontal alpha asymmetry, phase locking value, and any Tattva flags that fired.

## How it works

Each epoch is bandpassed 0.5 to 50 Hz, notched at 50 Hz for mains hum, and put
through a Welch PSD with half-second segments. Band powers come out of that as
fractions of total power.

Two decisions in here are worth explaining.

**Band edges follow the person, not the textbook.** Everyone's alpha rhythm
peaks in a slightly different place, anywhere from about 7.5 to 12.5 Hz. Fixed
8 to 13 Hz bounds therefore leak alpha into theta for some people and beta into
alpha for others. The extractor finds each epoch's own alpha peak and anchors
the theta, alpha and low-beta edges to it, following Klimesch (1999). If no
clear peak is present it falls back to 10 Hz and says so in `iaf_detected`.

**The classifier is rules, not a model.** There was a Random Forest here at
first, trained on synthetic data from `data_generator.py`. Synthetic data
cannot teach a model what a real Muse signal looks like, so it was learning the
generator rather than the phenomenon. It got replaced with fuzzy scoring over
explicit thresholds, which has the practical advantage that a wrong answer
points straight at the rule that produced it. It also starts instantly instead
of training on boot.

Beta is split into low (13 to 18 Hz) and high (18 to 30 Hz). That one change
fixed the biggest early bug: frontal sensors sitting on the forehead always
read elevated total beta, which is just normal waking physiology, so
everything came back as 99 percent Rajas. Only high beta drives that score now.

Artifact screening is a real gate rather than a warning flag. Blinks, jaw
clenches, flatlined electrodes and spiky non-Gaussian noise get the epoch
rejected with a 422 and an explanation, instead of being confidently classified
as a meditative state.

Two optional request fields exist because per-epoch classification in isolation
was too jumpy. `baseline` accepts per-band mean and standard deviation from a
short resting calibration, so that a person whose resting gamma naturally sits
high is not permanently reported as transcendent. `recent_probs` accepts the
last few epochs' probability dicts and blends them with an exponential moving
average, which stops the displayed state flickering every two seconds on one
noisy window.

## Running it

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

That starts the dev server on port 5000. Production uses gunicorn with a single
worker, which is what `render.yaml` sets up:

```bash
gunicorn -w 1 -b 0.0.0.0:$PORT "neuro_yogic.main:app"
```

Environment variables: `PORT`, and `CORS_ORIGINS` if you want to stop allowing
every origin, which is the default.

Quick check that it is alive:

```bash
curl -X POST localhost:5000/analyze/bands \
  -H 'Content-Type: application/json' \
  -d '{"delta":0.2,"theta":0.25,"alpha":0.3,"beta":0.15,"gamma":0.1}'
```

## Current state

It works and is deployed, and it is the version EEG-UI talks to. Some honest
caveats:

- The Guna mapping is a heuristic I put together, not a validated measure. The
  actual published Guna research is questionnaire-based personality work; there
  is no established EEG signature for Rajas the way there is for a sleep
  spindle. It is reasonable biofeedback framing and nothing more. The module
  docstring says the same thing.
- There are no tests. Thresholds were tuned by watching real sessions and
  adjusting, which is not the same as validation.
- `data_generator.py` and the joblib save/load paths on `YogaClassifier` are
  leftovers from the Random Forest version. Nothing in the request path uses
  them.
- `eeg_streamer.py` is a BrainFlow wrapper for reading a headband directly over
  USB or BLE from Python. BrainFlow is not in `requirements.txt` and the API
  never imports it, so treat that file as unused unless you install BrainFlow
  yourself.
- Single gunicorn worker on the free Render tier means the first request after
  a quiet period is slow while the instance wakes.

## Tech

Flask, flask-cors, NumPy, SciPy for the DSP, gunicorn in production, deployed on
Render. scikit-learn, pandas and joblib are still in the requirements from the
Random Forest era.

MIT licensed, see LICENSE.
