"""Optional local HTTP facade; pipelines are injected rather than built per request."""
from threading import Lock
from .routing import Router
from .storage import IntegrityError


def create_app(namespaces):
    from fastapi import FastAPI, HTTPException
    app = FastAPI(title="Vector Graph RAG Bilingualism", version="0.1.0")
    routers = {name: Router(pipelines) for name, pipelines in namespaces.items()}
    locks = {name: Lock() for name in namespaces}

    @app.get("/health")
    def health():
        return {"status": "ok", "namespaces": {name: list(router.pipelines) for name, router in routers.items()}}

    @app.post("/v1/query")
    def query(body: dict):
        namespace, question = body.get("namespace"), body.get("question")
        if not isinstance(namespace, str) or namespace not in routers or not isinstance(question, str) or not question.strip():
            raise HTTPException(400, "Valid namespace and nonempty question required")
        language = body.get("language", "auto")
        if language not in ("zh", "en", "auto", "both"):
            raise HTTPException(400, "Invalid language")
        try:
            with locks[namespace]:
                return routers[namespace].query(question, language=language).as_dict()
        except IntegrityError:
            raise HTTPException(409, "Index integrity validation failed") from None
        except NotImplementedError:
            raise HTTPException(400, "Joint-language mode is not implemented") from None
        except ValueError:
            raise HTTPException(400, "Pipeline configuration or request mismatch") from None
        except Exception:
            raise HTTPException(502, "Pipeline execution failed") from None

    return app
