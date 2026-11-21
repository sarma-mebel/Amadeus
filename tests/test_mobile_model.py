"""Mobile model selection is scoped to one authenticated turn."""
import httpx
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from amadeus.config import Config
from amadeus.providers.llm.openai_provider import OpenAIProvider


@pytest.mark.asyncio
async def test_selected_model_does_not_change_other_turns(monkeypatch, tmp_path):
    import amadeus.web.app as app_module

    config = Config()
    config.web.api_token = "local-fixture"
    config.web.demo_mode = False
    config.audio.enable_voice = False
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model = "gpt-4o-mini"
    complete = Mock(return_value=SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content="Я слушаю."))
    ]))
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    monkeypatch.setattr(app_module, "load_llm", lambda config: provider)
    app = app_module.create_app(config)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {"message": "Первый вопрос", "user_id": "qa-owner", "audio_enabled": False}
        assert (await client.post("/api/chat/mobile", json=payload)).status_code == 401
        client.headers["Authorization"] = "Bearer local-fixture"
        assert (await client.post("/api/chat/mobile?model=bad", json=payload)).status_code == 422
        response = await client.post("/api/chat/mobile", json=payload)
        assert response.status_code == 200, response.text
        assert any(call.kwargs.get("model") == "gpt-6.2-astra" and call.kwargs.get("reasoning_effort") == "medium" for call in complete.call_args_list)
        assert provider.model == "gpt-4o-mini"
        complete.reset_mock()
        payload.update(session_id=response.json()["session_id"], message="Следующий вопрос")
        assert (await client.post("/api/chat", json=payload)).status_code == 200
        assert complete.called
        assert all(call.kwargs["model"] == "gpt-4o-mini" for call in complete.call_args_list)
