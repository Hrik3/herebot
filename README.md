# Herebot

Herebot is a curiosity-driven learning engine. It runs as an interactive
command-line session: you type input, and it responds with curiosity
artifacts such as questions, hypotheses and reasoning outputs. It keeps
persistent state between runs.

## Requirements

- Python 3.9 or newer
- The following Python packages:
  `numpy`, `scipy`, `psutil`, `matplotlib`, `networkx`, `pandas`, `scikit-learn`

## Setup

```bash
git clone https://github.com/Hrik3/herebot.git
cd herebot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy scipy psutil matplotlib networkx pandas scikit-learn
```

## Running

Start the engine from the repository root:

```bash
python herebot/main.py
```

You will get a `Herebot>` prompt. Type a message and press Enter.
Type `exit` or `quit` (or press Ctrl+C) to shut down. On shutdown the engine
saves its state and writes its tracking plots.

## Abbreviations

Hardcoded response templates use the `TPL.` prefix (for example
`TPL.ASK.DEF`). The full list of template abbreviations is in
[`abbreavations.md`](abbreavations.md) at the repository root. Note that the
file name is spelled `abbreavations.md`.

## Repository layout

| Path | Contents |
|------|----------|
| `herebot/main.py` | Entry point |
| `herebot/` | Engine source (core, components, traits, concepts, backend) |
| `abbreavations.md` | Template abbreviation reference |
| `doc/`, `docs/` | Component and backend documentation |
| `plan.md`, `research.md`, `behaviour.md` | Design notes |
| `changelog.txt` | Change history |

## License

Herebot is licensed under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE) for the full text.
