# GLM-5.3 Flash NVFP4 endpoint image

Builds the model author's pinned `dev/jovian-judgement` vLLM branch for a
private Hugging Face Inference Endpoint on four RTX PRO 6000 Blackwell GPUs.

The endpoint contract is in `endpoint-config.json`. GitHub Actions tests the
contract, builds the upstream `vllm-openai` Docker target for SM120, and
publishes an immutable public image to GHCR.

## Verify locally

```sh
python3 -m unittest discover -s tests -v
```

The endpoint uses vLLM's OpenAI-compatible API on port 8000 and its native
`/health` readiness route. Scale-to-zero is set to 15 minutes to limit idle
cost; a running replica costs $11/hour.
