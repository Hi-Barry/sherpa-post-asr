"""
API 数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional


class TranscribeRequest(BaseModel):
    """上传音频转写请求"""
    audio_path: Optional[str] = Field(None, description="音频文件路径（服务端本地路径）")


class DiffItem(BaseModel):
    """纠错对比项"""
    position: int = Field(..., description="字符位置（相对于原文本）")
    original: str = Field(..., description="原始识别文字")
    corrected: str = Field(..., description="纠错后文字")
    confidence: float = Field(..., description="原始置信度")
    context_before: str = Field("", description="上文")
    context_after: str = Field("", description="下文")


class CorrectionSegment(BaseModel):
    """纠错段详情"""
    start: int = Field(..., description="起始字符位置")
    end: int = Field(..., description="结束字符位置")
    original: str = Field(..., description="原始文本段")
    corrected: str = Field(..., description="纠错后文本段")
    avg_confidence: float = Field(..., description="段内平均置信度")
    reason: str = Field("", description="纠错原因")


class TranscribeResponse(BaseModel):
    """转写响应"""
    raw_text: str = Field(..., description="原始 SenseVoice 识别文本")
    corrected_text: str = Field(..., description="局部纠错后文本")
    diffs: list[DiffItem] = Field(default_factory=list, description="逐字对比差异")
    segments: list[CorrectionSegment] = Field(
        default_factory=list, description="纠错段详情"
    )
    latency_ms: dict[str, float] = Field(
        default_factory=dict, description="各阶段延迟(ms)"
    )
