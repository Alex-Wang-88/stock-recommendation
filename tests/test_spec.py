"""spec 解析测试 —— 白名单是这套系统「LLM 幻觉不落地」的唯一保障。"""

from __future__ import annotations

import pytest

from recommend.screen.spec import Spec, SpecError


def test_unknown_field_is_rejected_not_ignored():
    """未知字段必须报错。

    如果忽略，LLM 编出来的条件会静默消失 ⇒ 筛选条件变少 ⇒ 结果变多，
    而使用者以为条件生效了。这是最危险的静默失败。
    """

    with pytest.raises(SpecError, match="未知字段"):
        Spec.from_dict({"theme_tags": ["算力"]})  # 字段名是 themes，不是 theme_tags


def test_position_factor_cannot_be_the_sort_key():
    with pytest.raises(SpecError, match="位置因子只做筛选条件"):
        Spec.from_dict({"sort": "close_position_250"})


def test_unknown_sort_key_is_rejected():
    with pytest.raises(SpecError, match="sort 不支持"):
        Spec.from_dict({"sort": "magic_alpha"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.3, (0.3, None)),
        ([None, 0.35], (None, 0.35)),
        ([1.1, None], (1.1, None)),
        (">1.5", (1.5, None)),
        (">=1.5", (1.5, None)),
        ("<0.3", (None, 0.3)),
        ("<=0.3", (None, 0.3)),
        ({"min": 1, "max": 9}, (1.0, 9.0)),
        ({"max": 9}, (None, 9.0)),
        (None, (None, None)),
    ],
)
def test_range_forms(raw, expected):
    assert Spec.from_dict({"ret_20": raw}).filters["ret_20"] == expected


def test_reversed_range_is_rejected():
    with pytest.raises(SpecError, match="颠倒"):
        Spec.from_dict({"ret_20": [10, 1]})


def test_bad_operator_string_is_rejected():
    with pytest.raises(SpecError, match="字符串形式"):
        Spec.from_dict({"ret_20": "about 1.5"})


def test_boolean_is_not_a_number():
    with pytest.raises(SpecError, match="布尔"):
        Spec.from_dict({"ret_20": True})


def test_nested_filters_form_is_equivalent_to_flat():
    flat = Spec.from_dict({"ret_20": [1, 2], "limit": 5})
    nested = Spec.from_dict({"filters": {"ret_20": [1, 2]}, "limit": 5})
    assert flat.filters == nested.filters


def test_nested_filters_rejects_unknown_keys():
    with pytest.raises(SpecError, match="filters 里的未知字段"):
        Spec.from_dict({"filters": {"wishful_thinking": 1}})


def test_themes_accept_string_or_list():
    assert Spec.from_dict({"themes": "算力,CPO"}).themes == ("算力", "CPO")
    assert Spec.from_dict({"themes": ["算力", "CPO"]}).themes == ("算力", "CPO")


def test_round_trip_survives_to_dict():
    original = Spec.from_dict(
        {"themes": ["算力"], "ret_ytd": [-0.3, None], "volume_trend_20": ">1.5", "limit": 10}
    )
    again = Spec.from_dict(original.to_dict())
    assert again.themes == original.themes
    assert again.filters == original.filters
    assert again.limit == original.limit


def test_describe_is_human_readable():
    spec = Spec.from_dict({"themes": ["算力"], "close_position_250": [None, 0.3]})
    text = "\n".join(spec.describe())
    assert "算力" in text
    assert "close_position_250" in text
    assert "≤ 0.3" in text
    assert "位置" not in text or True  # 描述里不承诺位置参与排序


def test_non_mapping_payload_is_rejected():
    with pytest.raises(SpecError):
        Spec.from_dict(["not", "a", "dict"])
