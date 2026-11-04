# Security and private data

Report suspected security issues privately to **wangkunze@qnbot.com**. Include the affected revision and a minimal reproduction without participant recordings, credentials or personal calibration values.

Only process trusted recordings and checkpoints. Pickle/PyTorch model loading and native SDKs can execute code; a matching hash proves identity, not safety. Do not expose live-preview or job-service endpoints to untrusted networks; configure host access controls and platform authentication. Keep capture devices under one owning process.

The repository must not contain access tokens, SSH passwords/keys, private model files or identifiable raw recordings. Deployment bundles and source releases have different distribution scopes; see THIRD_PARTY_NOTICES.md.
