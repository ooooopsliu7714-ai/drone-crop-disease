"""FastAPI 算法服务：对队友暴露 `POST /algo/predict`（字段严格遵循 Apifox 契约）。

启动（Windows，在项目根目录）：
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

接口一览：
    GET  /health                健康检查（不鉴权，给运维/联调用）
    POST /algo/predict          一个地址同时支持两种请求方式（见下）
    POST /algo/upload           先上传拿 image_id（可选，队友 Java 侧一般自己存图）
    GET  /algo/result/{image_id} 复查上次结果（对应小程序"看结果"）

为什么 /algo/predict 要自己写路由分发：
Apifox 里队友写的是"image_id 或直接上传图片文件"两种方式共存一个地址，
FastAPI 原生做不到同一路径既吃 JSON 又吃 multipart，所以用自定义 APIRoute：
先按 JSON 解析（拿到 image_id 就按 id 处理），否则退化成 multipart 读文件。
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from src.api.schema import HealthResponse, PredictRequest, PredictResponse
from src.api.service import AlgorithmService
from src.runtime.registry import (
    UPLOADS_DIR,
    build_predictor,
    cache_result,
    get_cached_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.0"
# 单文件上传上限（无人机正射图可能很大，这里给到 200MB）
MAX_UPLOAD_BYTES = int(os.environ.get("ALGO_MAX_UPLOAD_MB", "200")) * 1024 * 1024
CHUNK = 1024 * 1024


# ---------------------------------------------------------------------------
# 服务单例：进程启动时构造一次，避免每个请求都加载模型
# ---------------------------------------------------------------------------
_PREDICTOR, _STATUS = build_predictor()
SERVICE = AlgorithmService(_PREDICTOR, _STATUS)


class PredictRoute(APIRoute):
    """自定义路由：同一路径同时支持 JSON(image_id) 与 multipart(file)。

    实现要点：只在请求体是「JSON 且带 image_id」或 multiline 的 multipart 时接管，
    其余情况一律把请求交回 FastAPI 原生处理器（原处理器负责依赖注入与响应模型校验）。
    直接替换原生处理器会让 FastAPI 把响应对象当 ASGI app 调用，从而
    报 "'PredictResponse' object is not callable" 或让 /health 之类的路由莫名 422。
    """

    def get_route_handler(self) -> Callable[[Request], Any]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Any:
            content_type = (request.headers.get("content-type") or "").lower()

            if content_type.startswith("application/json"):
                # 只有"用 image_id 查图"这一种 JSON 语义需要接管；
                # 其它 JSON（未来扩展的字段）仍交给原处理器。
                raw = await request.body()
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    raise HTTPException(status_code=400, detail="请求体不是合法 JSON")
                image_id = (payload or {}).get("image_id")
                if image_id:
                    return _as_response(await _run_image_id(str(image_id), (payload or {}).get("crop")), self)
                # 没有 image_id：JSON 走原生校验（会返回 422 并说明缺哪个字段）

            if content_type.startswith("multipart/form-data"):
                form = await request.form()
                upload = form.get("file") or form.get("image") or form.get("img")
                if upload is not None and hasattr(upload, "read"):
                    data = await _read_upload(upload)  # type: ignore[arg-type]
                    resp = await _run_bytes(data, getattr(upload, "filename", "") or "", form.get("crop"))
                    return _as_response(resp, self)
                # 没有文件：交给原生处理器给出统一的校验错误

            return await original(request)

        return handler


def _as_response(payload: Any, route: APIRoute) -> JSONResponse:
    """把 pydantic 对象序列化成 JSONResponse。

    为什么必须手动做：自定义 route handler 绕过了 FastAPI 的 serialization 步骤，
    直接 return 一个 pydantic 对象会让 Starlette 把它当 ASGI app 调用，
    报 "'PredictResponse' object is not callable"。这里显式用路由声明的
    response_model 校验并序列化，保证响应仍受契约约束（多余字段会被剔除）。
    """
    model = getattr(route, "response_model", None)
    if model is not None and hasattr(model, "model_validate"):
        payload = model.model_validate(payload)  # type: ignore[attr-defined]
        return JSONResponse(content=payload.model_dump())  # type: ignore[attr-defined]
    if hasattr(payload, "model_dump"):
        return JSONResponse(content=payload.model_dump())
    if isinstance(payload, dict):
        return JSONResponse(content=payload)
    return JSONResponse(content={"code": 200, "message": "success", "data": str(payload)})


# ---------------------------------------------------------------------------
# 具体执行（放线程池，避免阻塞事件循环）
# ---------------------------------------------------------------------------
async def _read_upload(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        part = await upload.read(CHUNK)
        if not part:
            break
        total += len(part)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"图片超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限，请先压缩或改用切片上传",
            )
        chunks.append(part)
    return b"".join(chunks)


async def _run_bytes(data: bytes, filename: str, crop: str | None) -> PredictResponse:
    def _work() -> PredictResponse:
        resp = SERVICE.predict_file(data, filename=filename, crop=crop)
        cache_result(resp.image_id, resp.model_dump())
        return resp

    try:
        return await asyncio.to_thread(_work)
    except ValueError as exc:
        # 图片解码失败等客户端问题
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - 兜底，保证不返 500 白屏
        raise HTTPException(status_code=500, detail=f"算法推理失败：{exc}") from exc


async def _run_image_id(image_id: str, crop: str | None) -> PredictResponse:
    cached = get_cached_result(image_id)
    if cached and cached.get("code") == 200:
        # 同一个 image_id 重复请求直接回放，联调更省时间
        cached = dict(cached)
        cached["message"] = "success (cached)"
        return PredictResponse.model_validate(cached)

    def _work() -> PredictResponse:
        resp = SERVICE.predict_by_id(image_id, crop=crop)
        cache_result(resp.image_id, resp.model_dump())
        return resp

    try:
        return await asyncio.to_thread(_work)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class LabelRouter(APIRouter):
    """把本模块所有路由都用 PredictRoute 构建，从而拿到自定义分发能力。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("route_class", PredictRoute)
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
router = LabelRouter()


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        backend=_STATUS.backend,
        model_loaded=_STATUS.loaded,
        version=VERSION,
        device=_STATUS.device,
    )


@router.post(
    "/algo/predict",
    response_model=PredictResponse,
    summary="病害识别（Apifox 契约主接口）",
    description=(
        "两种调用方式：\n"
        "1) JSON: {\"image_id\":\"abc123\"}\n"
        "2) multipart/form-data: file=<图片>\n"
        "返回 disease_name / confidence / boxes 等字段，与 Apifox 完全一致。"
    ),
)
async def predict(payload: PredictRequest | None = None) -> PredictResponse:
    """处理 JSON 形态（image_id 方式）。

    文件上传形态由 PredictRoute 在进入本函数前直接接管，不会走到这里；
    payload 为 None 说明请求体是 multipart 但缺少 file 字段，给出明确提示。
    """
    if payload is None or not payload.image_id:
        raise HTTPException(
            status_code=422,
            detail=(
                "缺少参数：JSON 请求需提供 image_id（如 {\"image_id\":\"abc123\"}）；"
                "上传图片请用 multipart/form-data 并带 file 字段"
            ),
        )
    return await _run_image_id(payload.image_id, payload.crop)


@router.post("/algo/upload", summary="上传图片，返回 image_id")
async def upload(file: UploadFile = File(..., description="无人机航拍图或手机拍摄图")) -> dict[str, Any]:
    data = await _read_upload(file)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    image_id = f"{int(time.time() * 1000)}_{os.urandom(2).hex()}"
    dest = UPLOADS_DIR / f"{image_id}{suffix}"
    dest.write_bytes(data)
    return {
        "code": 200,
        "message": "success",
        "image_id": f"{image_id}{suffix}",
        "size_bytes": len(data),
        "path": str(dest),
    }


@router.get("/algo/result/{image_id}", summary="查询历史结果")
async def result(image_id: str) -> dict[str, Any]:
    cached = get_cached_result(image_id)
    if cached is None:
        raise HTTPException(status_code=404, detail=f"没有 image_id={image_id} 的历史结果")
    return cached


@router.get("/algo/labels", summary="标签体系（给前端做下拉/字典）")
async def labels() -> dict[str, Any]:
    from src.labels import LABELS

    return {
        "code": 200,
        "classes": LABELS.classes,
        "severity_levels": LABELS.severity_levels,
    }


# ---------------------------------------------------------------------------
# 应用装配：把 /algo/predict 换成自定义路由类
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="无人机农作物病害识别算法服务",
        description="RT-DETR + DINOv3 + SAM 2.1 三级流水线的 HTTP 接口层",
        version=VERSION,
    )
    app.include_router(router)

    @app.on_event("startup")
    async def _startup() -> None:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        print(
            f"[algo] 启动完成 backend={_STATUS.backend} loaded={_STATUS.loaded} "
            f"device={_STATUS.device} version={VERSION}"
        )
        for note in _STATUS.notes:
            print(f"[algo] 注意：{note}")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": f"服务端异常：{exc}", "image_id": ""},
        )

    return app


app = create_app()


if __name__ == "__main__":  # 便于 python -m src.api.app 直接起服务
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
