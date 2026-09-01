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
        self.assertEqual(
            config["container"]["image"],
            "ghcr.io/lanceyvang/glm53-flash-endpoint:0b67266a0f37d6146a8403fb8482403c62f412d5-b12x-12aea7d",
        )
        self.assertEqual(config["container"]["args"][0], "/repository")
        self.assertNotIn("--revision=378ca54585c46542bad1f3cb3ed0d73ae51cdb62", config["container"]["args"])
        self.assertIn("--tensor-parallel-size=4", config["container"]["args"])
        self.assertIn("--max-model-len=32768", config["container"]["args"])
        self.assertIn("--block-size=256", config["container"]["args"])
        self.assertIn("--attention-backend=B12X", config["container"]["args"])
        self.assertEqual(config["scaling"], {"min_replica": 0, "max_replica": 1, "scale_to_zero_timeout": 15})
        self.assertEqual(config["authentication"], "private")

    def test_publish_workflow_layers_the_pinned_b12x_runtime(self):
        workflow = (ROOT / ".github/workflows/publish.yml").read_text()
        dockerfile = (ROOT / "Dockerfile.runtime").read_text()

        self.assertIn("0b67266a0f37d6146a8403fb8482403c62f412d5", workflow)
        self.assertIn("B12X_COMMIT: 12aea7d96928f540a64259e4e24ef7688093b515", workflow)
        self.assertIn("B12X_TAG: 12aea7d", workflow)
        self.assertIn("context: .", workflow)
        self.assertIn("file: ./Dockerfile.runtime", workflow)
        self.assertNotIn("cache-to: type=gha", workflow)
        self.assertIn("platforms: linux/amd64", workflow)
        self.assertIn("ghcr.io/lanceyvang/glm53-flash-endpoint", workflow)
        self.assertIn(
            "FROM ghcr.io/lanceyvang/glm53-flash-endpoint:0b67266a0f37d6146a8403fb8482403c62f412d5",
            dockerfile,
        )
        self.assertIn(
            'ARG B12X_COMMIT="12aea7d96928f540a64259e4e24ef7688093b515"',
            dockerfile,
        )
        self.assertIn("b12x/archive/${B12X_COMMIT}.tar.gz", dockerfile)


if __name__ == "__main__":
    unittest.main()
