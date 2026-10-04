# Jev access

The experiments query TypeSafe's Jev ("System One") through `typesafe-sdk`. The whole study
makes **at most 60,000 Jev inferences**, counted in a persistent ledger. Read
[the call budget](#the-inference-cap) before you run anything live.

## 1. Get access and an API key

1. Request early access at console.typesafe.ai. Jev is also reachable through OpenRouter,
   but this project calls the TypeSafe API directly.
2. Once you have access, create an API key in the console.

## 2. Configure the key

Put the key in a `.env` file at the project root. The file is gitignored; keep it private:

```bash
echo 'TYPESAFE_API_KEY=<your key>' > .env
chmod 600 .env
```

Alternatively `export TYPESAFE_API_KEY=...`; an exported variable takes precedence over
`.env`. The key is passed only to the SDK client. It never appears in the cache, the
request log, results or logs, and error messages show at most its last 4 characters.

Without a key, everything still runs in mock mode (`--mode mock`, the default), using
clearly labelled synthetic stand-ins for Jev and for the baseline.

## 3. Check connectivity

```bash
make setup
uv run python -m ciayn smoke --mode live --dry-run   # reports the 1 request it would make
make smoke
```

`make smoke` makes **one** Jev Choice query on a German Credit example and prints the full
class distribution, the requested and reported model version, the token count and latency.
If the key is missing it stops before any request and points back to this guide. If the API
reports a model other than the one pinned in `configs/jev.yaml`, it stops with an error so
that you can decide whether to continue on that version.

## The inference cap

`configs/budgets.yaml` caps Jev at `jev_total: 60000` requests for the whole study and
allocates them per stage. The 100 requests made under the earlier limit are in the ledger
and count.

| Stage | Budget | What it is for |
|---|---|---|
| `smoke` | 5 | connectivity and response-shape check (1 already spent) |
| `template` | 60 | 20 training-split examples for each text dataset, before templates are frozen |
| `predict` | 13,700 | 13,621 predictions (99 already spent): all 1,000 German Credit rows, all 569 breast-cancer rows, SST-2 validation 872 plus 500 threshold-selection items, AG News test 7,600, and the test items of Banking77 (3,080), a dataset since removed from the study |
| `explain` | 44,000 | KernelSHAP, LIME and the deletion test on 20 examples per dataset |
| all others | 0 | they read saved predictions only |

Every Jev request is recorded in a ledger inside `cache/responses.sqlite` *before* it is
sent, whether or not it succeeds, and SDK retries are disabled. Stage budgets and the global
cap are both checked against that ledger, so they hold across separate runs and crashes. A
stage that would go over stops before making any request and reports the requests required
and allowed.

- Always look at the dry-run report first, for example
  `uv run python -m ciayn predict --mode live --dry-run`.
- Reruns are served from the cache and cost nothing.
- A failed request still counts. The item is not cached, so rerunning the stage retries it
  while the budget has headroom; otherwise it counts as an error in every metric.
- Check the ledger at any time:
  `uv run python -c "from ciayn.backends.budget import Ledger; l = Ledger(); print(l.total(), 'of', l.cap)"`

## Order of the live stages

```bash
uv run python -m ciayn freeze                       # once; refuses to overwrite an existing freeze
uv run python -m ciayn template --mode live --dry-run
uv run python -m ciayn template --mode live         # up to 60 requests, then templates are frozen
uv run python -m ciayn predict --mode live --dry-run
uv run python -m ciayn predict --mode live
```

`freeze` writes the evaluation identifiers to `results/splits/` with a hash manifest; every
other stage checks it and stops if the identifiers changed. `predict` refuses to run for a
dataset without a frozen template, and sends the first request of each dataset alone so that
a wrong response shape stops the stage before the rest is spent.

## Request and response log

Every request is written to `cache/exchanges/<stage>.jsonl` as it crosses the wire: the
request body as sent, the response body as received, the SDK version, the requested and
reported model, the ledger number, a timestamp and the latency. Headers are not recorded.
Nothing is ever added for a call that was not logged when it happened.

The 100 calls made before this log existed have no records. `uv run python -m ciayn
payload-audit --mode live` regenerates the 99 cached requests from code, reports how many
hash to the stored cache keys, and writes the result to `results/live/payload_audit.json`.

## Backup

After every live stage the response cache, the ledger and the request log are copied
outside the working copy: to the project's S3 bucket once it exists, and to
`../jev_paper_backup/cache-latest/` until then. `uv run python -m ciayn backup --mode live`
does the same on demand. Deleting `cache/responses.sqlite` deletes the ledger; don't, unless
you mean to start the study again.

## Estimated cost

Input is billed at $0.042 per million tokens and output is free. A German Credit request is
about 520 input tokens; a request with many label descriptions is several times larger
(Banking77, since removed from the study, carried 77). The planned 57,765 requests are estimated at about 35 million tokens,
about **$1.50**. The actual cost is computed from the logged token counts and reported in
the paper.
