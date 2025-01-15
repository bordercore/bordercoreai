import pytest
from modules.util import sort_models


def test_sort_models():

    assert sort_models(
        [{"name": "b"}, {"name": "d"}, {"name": "e"}, {"name": "a"}],
        ["a", "d"]
    ) == [{"name": "a"}, {"name": "d"}, {"name": "b"}, {"name": "e"}]


@pytest.fixture
def strip_code_fences():
    def _strip_code_fences(text):
        lines = text.strip().split('\n')
        if lines[0].startswith('```') and lines[-1].startswith('```'):
            return '\n'.join(lines[1:-1])
        return text
    return _strip_code_fences


def test_basic_json_fence(strip_code_fences):
    input_text = """```json
{"category":"other"}
```"""
    expected = '{"category":"other"}'
    assert strip_code_fences(input_text) == expected


def test_multiline_content(strip_code_fences):
    input_text = """```python
def hello():
    print("Hello")
    return True
```"""
    expected = """def hello():
    print("Hello")
    return True"""
    assert strip_code_fences(input_text) == expected


def test_no_fence(strip_code_fences):
    input_text = "plain text\nno fences here"
    assert strip_code_fences(input_text) == input_text


def test_only_start_fence(strip_code_fences):
    input_text = """```python
some code
more code"""
    assert strip_code_fences(input_text) == input_text


def test_only_end_fence(strip_code_fences):
    input_text = """some code
more code
```"""
    assert strip_code_fences(input_text) == input_text


def test_empty_content(strip_code_fences):
    input_text = """```
```"""
    expected = ''
    assert strip_code_fences(input_text) == expected


def test_whitespace_handling(strip_code_fences):
    input_text = """```
   spaces before
spaces after   \n```"""
    expected = """   spaces before
spaces after   """
    assert strip_code_fences(input_text) == expected


def test_fence_with_text_after(strip_code_fences):
    input_text = """```python extra text
print("hello")
``` final text"""
    expected = 'print("hello")'
    assert strip_code_fences(input_text) == expected
