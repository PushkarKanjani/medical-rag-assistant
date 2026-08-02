from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from src.settings import AppSettings

router = APIRouter()


class AbhaCreateRequest(BaseModel):
    aadhaar_number: str
    mobile_number: str
    otp: str | None = None


class AbhaLoginRequest(BaseModel):
    abha_id: str
    mobile_number: str
    otp: str | None = None


class AbhaResponse(BaseModel):
    success: bool
    data: dict[str, Any]


async def _proxy_to_abdm(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    settings = AppSettings()
    async with httpx.AsyncClient(base_url=settings.abdm_sandbox_base_url, timeout=30.0) as client:
        response = await client.post(path, json=payload)
        response.raise_for_status()
        return response.json()


@router.post("/v1/abha/create", response_model=AbhaResponse)
async def create_abha(req: AbhaCreateRequest) -> AbhaResponse:
    init_payload = {"aadhaar_number": req.aadhaar_number, "mobile_number": req.mobile_number}
    init_response = await _proxy_to_abdm("v1/registration/aadhaar/generateOtp", init_payload)
    verify_payload = {"aadhaar_number": req.aadhaar_number, "mobile_number": req.mobile_number, "otp": req.otp}
    verify_response = await _proxy_to_abdm("v1/registration/aadhaar/verifyOtp", verify_payload)
    return AbhaResponse(success=True, data={"generateOtp": init_response, "verifyOtp": verify_response})


@router.post("/v1/abha/login", response_model=AbhaResponse)
async def login_abha(req: AbhaLoginRequest) -> AbhaResponse:
    init_payload = {"abha_id": req.abha_id, "mobile_number": req.mobile_number}
    init_response = await _proxy_to_abdm("v1/phr/login/loginSendOtp", init_payload)
    verify_payload = {"abha_id": req.abha_id, "mobile_number": req.mobile_number, "otp": req.otp}
    verify_response = await _proxy_to_abdm("v1/phr/login/loginVerifyOtp", verify_payload)
    return AbhaResponse(success=True, data={"loginSendOtp": init_response, "loginVerifyOtp": verify_response})