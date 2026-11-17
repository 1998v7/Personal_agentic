import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import yaml

from IMAct.exec_layer.config import load_outer_config
from IMAct.models.base import ChatRequest
from IMAct.models.catalog import ModelCatalog, load_catalog
from IMAct.models.config import create_provider


def config():
    return {
        "roles": {"action": "qwen", "user": "qwen", "judge": "qwen"},
        "models": {"qwen": {"provider": "openai_compatible", "model": "Qwen3.8-27B",
                            "base_url": "https://antchat.alipay.com/v1/chat/completions",
                            "api_key_env": "TEST_ANTCHAT_KEY", "params": {"stream": False, "max_tokens": 4096}}},
    }


class CatalogTests(unittest.TestCase):
    def test_roles_unknown_keys_and_reserved_parameters_fail(self):
        for mutate in (
            lambda c: c["roles"].update(action="missing"),
            lambda c: c["models"]["qwen"].update(provder="typo"),
            lambda c: c["models"]["qwen"]["extra_body"].update(model="wrong"),
            lambda c: c["models"]["qwen"]["params"].update(stream=True),
        ):
            raw = config()
            raw["models"]["qwen"]["extra_body"] = {}
            mutate(raw)
            with self.assertRaises(ValueError):
                ModelCatalog.model_validate(raw)

    def test_relative_path_env_priority_and_no_unrelated_credentials(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"TEST_ANTCHAT_KEY": "shell-secret"}):
            root = Path(temp)
            (root / ".env").write_text("TEST_ANTCHAT_KEY=file-secret\n")
            raw = config()
            raw["models"]["unused"] = {"model": "unavailable", "api_key_env": "UNSET_UNUSED_KEY"}
            (root / "models.yaml").write_text(yaml.safe_dump(raw))
            (root / "run.yaml").write_text("models_file: models.yaml\n")
            cfg = load_outer_config(root / "run.yaml")
            self.assertEqual(cfg.model.resolve_api_key(), "shell-secret")
            self.assertEqual(cfg.model.model, "Qwen3.8-27B")
            self.assertEqual(cfg.models_file, str((root / "models.yaml").resolve()))
            catalog = load_catalog(root / "models.yaml")
            self.assertEqual(catalog.selected("action")[0], "qwen")
            (root / "run.yaml").write_text("models_file: nonexistent.yaml\n")
            cfg = load_outer_config(root / "run.yaml", models_file=root / "models.yaml")
            self.assertEqual(cfg.model.model, "Qwen3.8-27B")
            (root / "run.yaml").write_text("models_file: models.yaml\nmodel_call:\n  params: {max_tokens: 1}\n")
            with self.assertRaisesRegex(ValueError, "混用"):
                load_outer_config(root / "run.yaml")

    def test_snapshot_never_contains_credentials(self):
        raw = config()
        raw["models"]["qwen"].update(api_key="inline-secret", headers={"X-LY-TOKEN": "header-secret"},
                                    extra_body={"secret": "body-secret"},
                                    base_url="https://name:password@host/v1?key=query-secret")
        snapshot = json.dumps(ModelCatalog.model_validate(raw).snapshot())
        for secret in ("inline-secret", "header-secret", "body-secret", "query-secret", "password"):
            self.assertNotIn(secret, snapshot)
        self.assertIn("TEST_ANTCHAT_KEY", snapshot)
        self.assertIn("4096", snapshot)


class WireTests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_exact_endpoint_headers_body_and_response(self):
        raw = config()
        raw["models"]["qwen"]["headers_env"] = {"Custom-Header": "TEST_CUSTOM_HEADER"}
        cfg = ModelCatalog.model_validate(raw).selected("action")[1]
        cfg.options["merge_system_messages"] = True
        content = [{"type": "image_url", "image_url": "https://example.com/image.jpg"},
                   {"type": "text", "text": "描述图片"}]

        def handler(request):
            self.assertEqual(str(request.url), "https://antchat.alipay.com/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer key")
            self.assertEqual(request.headers["Custom-Header"], "header-value")
            self.assertNotIn("SOFA-TraceId", request.headers)
            body = json.loads(request.content)
            self.assertEqual(body["model"], "Qwen3.8-27B")
            self.assertIs(body["stream"], False)
            self.assertEqual(body["messages"][0], {"role": "system", "content": "contract\n\npolicy"})
            self.assertEqual(body["messages"][1]["content"], content)
            return httpx.Response(200, json={"id": "real-shaped-id", "model": body["model"], "choices": [
                {"finish_reason": "stop", "message": {"role": "assistant", "content": "正文"}}]})

        with patch.dict(os.environ, TEST_ANTCHAT_KEY="key", TEST_CUSTOM_HEADER="header-value"):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                provider = create_provider(cfg, client=client)
                messages = [{"role": "system", "content": "contract"}, {"role": "system", "content": "policy"},
                            {"role": "user", "content": content}]
                response = await provider.chat(ChatRequest(messages=messages, params=cfg.params))
                self.assertEqual(response.content, "正文")
                self.assertEqual(len(messages), 3)  # 不修改执行层保存的原始输入
