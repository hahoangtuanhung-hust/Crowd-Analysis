import pytest

from modal_common_path import _validate_cpu_request


@pytest.mark.parametrize("value", [1, 2, 4, 8])
def test_modal_cpu_request_accepts_supported_profiles(value: int) -> None:
    assert _validate_cpu_request(value) == float(value)


@pytest.mark.parametrize("value", [0.5, 3, 16])
def test_modal_cpu_request_rejects_unsafe_or_unprofiled_values(value: float) -> None:
    with pytest.raises(ValueError, match="cpu must be one of"):
        _validate_cpu_request(value)
