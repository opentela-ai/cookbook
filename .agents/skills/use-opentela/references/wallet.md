# Wallet & ownership

A Solana Ed25519 keypair is OpenTela's **ownership** identity: a short
Provider ID (`otela-<first 12 of pubkey>`) is written into the `owner` field
of every service the node registers, so the mesh can audit who runs what.
Canonical source: <https://opentela.ai/docs/tutorial/owner>.

The keypair is a real Solana wallet — importable into Phantom (base58 private
key), Solflare/Solana CLI (JSON keypair file), or MetaMask (Solana Snap).

## Daily commands

```bash
./otela init                         # first run: ~/.config/opentela/ + cfg.yaml + default wallet
./otela wallet create                # additional wallet (first in the list is the default/active one)
./otela wallet list                  # * marks the default wallet
./otela wallet info                   # default wallet pubkey, Provider ID, keypair path
./otela balance                       # SOL + OTELA (uses solana.mint from cfg.yaml); --solana.rpc for devnet
./otela wallet transfer <PUBKEY> <SOL>            # native SOL, amount in SOL not lamports
./otela wallet airdrop 2 --solana.rpc https://api.devnet.solana.com   # devnet/testnet only
```

Export / import:

```bash
./otela wallet export                          # base58 private key → paste into Phantom
./otela wallet export --file ~/key.json        # Solana-CLI JSON int-array → Solflare / solana CLI
./otela wallet import ~/existing-keypair.json  # copies it in; original untouched
```

## Storage

```
~/.config/opentela/                # config + wallets (all 0600)
├── cfg.yaml
├── accounts.json                  # registry of managed wallets
└── accounts/<pubkey>/keypair.json # Solana-CLI format (importable anywhere)
```

Node data (keys, CRDT DBs) lives separately under `~/.ocfcore/`. `otela init`
migrates any legacy `~/.ocf/` wallet on first access.

## Config & env

All wallet settings live in `~/.config/opentela/cfg.yaml`. CLI flags and env
vars (prefix **`OF_`**) override them — note `OF_*` are node-config vars,
distinct from `OTELA_API` which is the **public-gateway** bearer token.

| Key | CLI flag | Env var |
|---|---|---|
| `account.wallet` (keypair path) | `--account.wallet` | `OF_ACCOUNT_WALLET` |
| `wallet.account` (pubkey override) | `--wallet.account` | `OF_WALLET_ACCOUNT` |
| `solana.rpc` | `--solana.rpc` | `OF_SOLANA_RPC` |
| `solana.mint` | `--solana.mint` | `OF_SOLANA_MINT` |
| `solana.skip_verification` | `--solana.skip_verification` | `OF_SOLANA_SKIP_VERIFICATION` |

## Running without a wallet

`otela start --wallet.account ""` logs `Wallet account set to 'none'` and runs
normally for routing/serving — the `owner` field in the node table is just
empty. Fine for quick local testing.

## Why this matters for routing

The wallet is loaded **automatically** at `otela start`; you don't pass a flag.
If you run several workers from the **same** `~/.config/opentela/`, they all
share one Provider ID (one operator, many machines). Different operators get
different Provider IDs, so `owner` in `/v1/dnt/table` answers "who serves
this". A `X-Otela-Trust` header (see routing.md) lets a caller require
self-attested (`1`) or user-trusted (`2`) peers — and trust level is
**computed locally** from each head's `trusted_wallets`, not a network
consensus.
