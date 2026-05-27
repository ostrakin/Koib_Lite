# -*- coding: utf-8 -*-
"""
Koib-V-4.6 — Модуль генерации ответов
★ ИСПРАВЛЕНО: asyncio.Semaphore ограничивает параллельные генерации
  (защита от OOM при одновременных запросах от VK-бота)
★ Полностью асинхронный (aiohttp)
"""
import logging
from typing import List, Dict, Any, Optional
import aiohttp
import asyncio
import ssl

from .retrieval import RetrievalResult
from config import (
    LLM_PROVIDER, GIGACHAT_CREDENTIALS, GIGACHAT_MODEL,
    GIGACHAT_TEMPERATURE, GIGACHAT_MAX_TOKENS, GIGACHAT_TIMEOUT,
    GIGACHAT_VERIFY_SSL, OPENAI_API_KEY, OPENAI_LLM_MODEL,
    OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, LOCAL_LLM_MODEL, LOCAL_LLM_URL,
    MAX_CONCURRENT_GENERATIONS,
)

logger = logging.getLogger("koib.generation")

SYSTEM_PROMPT = """Ты — эксперт-ассистент по технической документации. Твоя задача — отвечать на вопросы пользователя строго на основе предоставленного контекста из документации.

ПРАВИЛА ОТВЕТА:
1. **Опирайся ТОЛЬКО на предоставленный контекст.** Не придумывай информацию, которой нет в контекстных фрагментах. Если контекст не содержит ответа — честно сообщи: «В предоставленной документации нет информации по этому вопросу.»
2. **Цитируй источники.** Каждое утверждение в ответе должно сопровождаться ссылкой на источник в формате: [Документ: {имя_файла}, стр. {номер}]. Если информация из нескольких источников — укажи все.
3. **Таблицы.** Если в контексте есть таблица и она релевантна вопросу, воспроизведи её в формате Markdown, затем прокомментируй данные.
4. **Формулы.** Если в контексте есть формулы, выведи их в формате LaTeX и объясни значение переменных. Если переменные не объяснены в контексте — укажи это.
5. **Схемы и рисунки.** Если контекст содержит описание рисунка или схемы, опиши его текстуально и укажи источник.
6. **Структура ответа.** Отвечай структурированно: используй заголовки, списки, выделение важного. Начинай с прямого ответа, затем давай пояснения.
7. **Не повторяй вопрос.** Переходи сразу к ответу.
8. **Язык.** Отвечай на том же языке, на котором задан вопрос (по умолчанию — русский)."""


def build_prompt(query: str, results: List[RetrievalResult]) -> str:
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"--- Фрагмент {i} ---")
        context_parts.append(r.to_context_string())
        context_parts.append("")
    context_text = "\n".join(context_parts)
    return (
        f"КОНТЕКСТ ИЗ ТЕХНИЧЕСКОЙ ДОКУМЕНТАЦИИ:\n{context_text}\n\n"
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{query}\n\n"
        f"Ответь на вопрос, строго опираясь на приведённый выше контекст. "
        f"Обязательно цитируй источники в формате [Документ: имя_файла, стр. X]."
    )


class LLMClient:
    """Полностью асинхронный клиент LLM."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or LLM_PROVIDER
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            ssl_ctx = None
            if not GIGACHAT_VERIFY_SSL and self.provider == "gigachat":
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = GIGACHAT_MAX_TOKENS,
        temperature: float = GIGACHAT_TEMPERATURE,
    ) -> str:
        sys_prompt = system_prompt or SYSTEM_PROMPT
        if self.provider == "gigachat":
            return await self._generate_gigachat_async(prompt, sys_prompt, max_tokens, temperature)
        elif self.provider == "openai":
            return await self._generate_openai_async(prompt, sys_prompt, max_tokens, temperature)
        elif self.provider == "local":
            return await self._generate_local_async(prompt, sys_prompt, max_tokens, temperature)
        return f"Провайдер '{self.provider}' не поддерживается."

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: int = GIGACHAT_MAX_TOKENS,
                 temperature: float = GIGACHAT_TEMPERATURE) -> str:
        """Синхронная обёртка."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run,
                        self.generate_async(prompt, system_prompt, max_tokens, temperature)
                    ).result()
            return loop.run_until_complete(
                self.generate_async(prompt, system_prompt, max_tokens, temperature)
            )
        except RuntimeError:
            return asyncio.run(self.generate_async(prompt, system_prompt, max_tokens, temperature))

    async def _generate_gigachat_async(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        if not GIGACHAT_CREDENTIALS:
            return "Ошибка: GIGACHAT_CREDENTIALS не заданы."
        session = await self._get_session()
        auth_headers = {
            "Authorization": f"Basic {GIGACHAT_CREDENTIALS}",
            "RqUID": "koib-rag-001",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            async with session.post(
                "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                headers=auth_headers,
                data={"scope": "GIGACHAT_API_PERS"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as auth_resp:
                if auth_resp.status != 200:
                    return f"Ошибка авторизации GigaChat: {auth_resp.status}"
                auth_data = await auth_resp.json()
                token = auth_data["access_token"]

            chat_headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            chat_payload = {
                "model": GIGACHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            async with session.post(
                "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                headers=chat_headers,
                json=chat_payload,
                timeout=aiohttp.ClientTimeout(total=GIGACHAT_TIMEOUT),
            ) as chat_resp:
                if chat_resp.status == 401:
                    async with session.post(
                        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                        headers=auth_headers,
                        data={"scope": "GIGACHAT_API_PERS"},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as reauth_resp:
                        if reauth_resp.status != 200:
                            return f"Ошибка повторной авторизации: {reauth_resp.status}"
                        reauth_data = await reauth_resp.json()
                        token = reauth_data["access_token"]
                    chat_headers["Authorization"] = f"Bearer {token}"
                    async with session.post(
                        "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                        headers=chat_headers,
                        json=chat_payload,
                        timeout=aiohttp.ClientTimeout(total=GIGACHAT_TIMEOUT),
                    ) as retry_resp:
                        if retry_resp.status != 200:
                            return f"Ошибка API GigaChat: {retry_resp.status}"
                        retry_data = await retry_resp.json()
                        return retry_data["choices"][0]["message"]["content"].strip()
                if chat_resp.status != 200:
                    return f"Ошибка API GigaChat: {chat_resp.status}"
                chat_data = await chat_resp.json()
                return chat_data["choices"][0]["message"]["content"].strip()
        except asyncio.TimeoutError:
            return "Таймаут запроса к GigaChat."
        except aiohttp.ClientError as e:
            return f"Ошибка соединения с GigaChat: {e}"
        except Exception as e:
            return f"Ошибка генерации GigaChat: {e}"

    async def _generate_openai_async(self, prompt, system_prompt, max_tokens, temperature):
        if not OPENAI_API_KEY:
            return "Ошибка: OPENAI_API_KEY не задан."
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model=OPENAI_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Ошибка OpenAI: {e}"

    async def _generate_local_async(self, prompt, system_prompt, max_tokens, temperature):
        try:
            session = await self._get_session()
            async with session.post(
                f"{LOCAL_LLM_URL}/api/generate",
                json={
                    "model": LOCAL_LLM_MODEL,
                    "prompt": f"{system_prompt}\n{prompt}",
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": temperature},
                },
                timeout=aiohttp.ClientTimeout(total=GIGACHAT_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return f"Ошибка локальной LLM: {resp.status}"
                data = await resp.json()
                return data.get("response", "").strip()
        except asyncio.TimeoutError:
            return "Таймаут локальной LLM."
        except Exception as e:
            return f"Ошибка локальной LLM: {e}"


class AnswerGenerator:
    """
    Полный RAG-пайплайн: поиск → промпт → LLM → валидация → лог.
    ★ Semaphore ограничивает параллельные генерации, предотвращая OOM.
    """

    def __init__(self):
        from .retrieval import HybridRetriever
        self.retriever = HybridRetriever()
        self.llm = LLMClient()
        # ★ КРИТИЧНО: жёсткий лимит параллельных генераций
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)

    async def answer_async(
        self,
        query: str,
        k: int = 4,
        model_filter: str = "",
        validate: bool = True,
    ) -> Dict[str, Any]:
        import time
        # ★ Семафор: если лимит исчерпан, запрос ждёт в очереди (без OOM-спайка)
        async with self._semaphore:
            t0 = time.time()
            results = self.retriever.search(query, k=k, model_filter=model_filter)

            if not results:
                answer_result = {
                    "answer": "По вашему запросу не найдено релевантных фрагментов в документации.",
                    "sources": [],
                    "results": [],
                    "context_text": "",
                    "validation": None,
                    "status": "review",
                }
                self._log_query(query, answer_result, model_filter)
                return answer_result

            prompt = build_prompt(query, results)
            answer = await self.llm.generate_async(prompt)

            validation_result = None
            if validate:
                try:
                    from .validation import AnswerValidator
                    validator = AnswerValidator()
                    validation_result = validator.validate(answer, results, query)
                    validation_dict = validation_result.to_dict()
                except Exception as exc:
                    logger.warning(f"Ошибка валидации: {exc}")
                    validation_dict = None
            else:
                validation_dict = None

            sources = [
                {"document": r.source, "page": r.page, "heading": r.heading,
                 "chunk_type": r.chunk_type, "score": r.score}
                for r in results
            ]

            status = "approved"
            final_answer = answer
            if validation_dict:
                if validation_dict.get("status") == "rejected":
                    status = "rejected"
                    final_answer = "По вашему запросу не найдено точного ответа в официальных источниках."
                elif validation_dict.get("status") == "review":
                    status = "review"

            answer_result = {
                "answer": final_answer,
                "sources": sources,
                "results": results,
                "context_text": prompt,
                "validation": validation_dict,
                "status": status,
                "latency": time.time() - t0,
            }
            self._log_query(query, answer_result, model_filter)
            logger.info(f"Ответ сгенерирован за {answer_result['latency']:.2f}с")
            return answer_result

    def answer(self, query: str, k: int = 4, model_filter: str = "",
               validate: bool = True) -> Dict[str, Any]:
        """Синхронная обёртка для CLI."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run,
                        self.answer_async(query, k, model_filter, validate)
                    ).result()
            return loop.run_until_complete(
                self.answer_async(query, k, model_filter, validate)
            )
        except RuntimeError:
            return asyncio.run(self.answer_async(query, k, model_filter, validate))

    def _log_query(self, query: str, result: Dict[str, Any], model_filter: str) -> None:
        try:
            from .logging_module import get_query_logger
            get_query_logger().log(
                query=query,
                answer=result.get("answer", ""),
                model_type=model_filter,
                sources=result.get("sources", []),
                validation_result=result.get("validation", {}),
                status=result.get("status", "unknown"),
                extra_metadata={
                    "num_chunks": len(result.get("results", [])),
                    "latency": result.get("latency", 0),
                },
            )
        except Exception as exc:
            logger.warning(f"Ошибка логирования: {exc}")