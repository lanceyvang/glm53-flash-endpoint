import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EndpointContractTest(unittest.TestCase):
    def test_endpoint_manifest_is_pinned_and_safe_by_default(self):
        config = json.loads((ROOT / "endpoint-config.json").read_text())

        self.assertEqual(
            config["model"],
            {
                "repository": "local-inference-lab/GLM-5.3-Flash-NVFP4",
                "revision": "378ca54585c46542bad1f3cb3ed0d73ae51cdb62",
            },
        )
        self.assertEqual(
            config["compute"],
            {
                "provider": "aws",
                "region": "us-east-2",
                "instance_type": "nvidia-rtx-pro-6000",
                "instance_size": "x4",
            },
        )
        self.assertEqual(config["container"]["port"], 8000)
        self.assertEqual(config["container"]["health_route"], "/health")
        self.assertEqual(config["container"]["args"][0], "/repository")
        self.assertNotIn("--revision=378ca54585c46542bad1f3cb3ed0d73ae51cdb62", config["container"]["args"])
        self.assertIn("--tensor-parallel-size=4", config["container"]["args"])
        self.assertIn("--max-model-len=32768", config["container"]["args"])
        self.assertEqual(config["scaling"], {"min_replica": 0, "max_replica": 1, "scale_to_zero_timeout": 15})
        self.assertEqual(config["authentication"], "private")

    def test_publish_workflow_builds_the_pinned_blackwell_target(self):
        workflow = (ROOT / ".github/workflows/publish.yml").read_text()
        blackwell_patch = (ROOT / "patches/blackwell-flash-attention.patch").read_text()

        self.assertIn("0b67266a0f37d6146a8403fb8482403c62f412d5", workflow)
        self.assertIn("repository: local-inference-lab/vllm", workflow)
        self.assertIn("fetch-depth: 0", workflow)
        self.assertIn("context: ./vllm", workflow)
        self.assertIn("name: Free runner disk for the CUDA build", workflow)
        self.assertIn("/usr/local/lib/android", workflow)
        self.assertIn("/opt/hostedtoolcache/CodeQL", workflow)
        self.assertIn("target: vllm-openai", workflow)
        self.assertIn("torch_cuda_arch_list=12.0", workflow)
        self.assertIn("patches/blackwell-flash-attention.patch", workflow)
        self.assertIn("set(FA3_ENABLED OFF)", blackwell_patch)
        self.assertIn("--- a/setup.py", blackwell_patch)
        self.assertIn(
            '-        ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))',
            blackwell_patch,
        )
        self.assertIn("platforms: linux/amd64", workflow)
        self.assertIn("ghcr.io/lanceyvang/glm53-flash-endpoint", workflow)


if __name__ == "__main__":
    unittest.main()
