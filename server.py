"""
API 服务启动入口 - 用于 Apifox 调试
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import router
from core.container import container


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理"""
    await container.startup()
    yield
    await container.shutdown()


app = FastAPI(
    title="农业气候与资源数据专家助手 API",
    description="基于 RAG 架构的农业领域智能问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "service": "agri-rag"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
