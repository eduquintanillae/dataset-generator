import pytest
from modules.dataset import DataLoader


@pytest.fixture
def data_loader():
    return DataLoader(
        file_paths=[
            "assets/attention_is_all_you_need.pdf",
            "assets/attention_is_all_you_need.docx",
            "assets/attention_is_all_you_need.txt",
        ]
    )


def test_read_pdf(data_loader):
    text = data_loader.read_pdf("assets/attention_is_all_you_need.pdf")
    assert text is not None
    assert len(text) > 0


def test_read_docx(data_loader):
    text = data_loader.read_docx("assets/attention_is_all_you_need.docx")
    assert text is not None
    assert len(text) > 0


def test_read_txt(data_loader):
    text = data_loader.read_txt("assets/attention_is_all_you_need.txt")
    assert text is not None
    assert len(text) > 0


def test_load_data(data_loader):
    data = data_loader.load_data()
    assert len(data) == 3
    assert all("file_path" in item and "content" in item for item in data)
    assert all(item["content"] is not None for item in data)
    assert all(item["content"] != "" for item in data)


def test_unsupported_file(data_loader):
    unsupported_file = "assets/unsupported_file.xyz"
    data_loader.file_paths.append(unsupported_file)
    data = data_loader.load_data()
    assert unsupported_file not in [item["file_path"] for item in data]
    assert len(data) == 3  # Original 3 files
    assert all(item["content"] is not None for item in data)


def test_empty_file_paths():
    empty_loader = DataLoader(file_paths=[])
    data = empty_loader.load_data()
    assert data == []
    assert len(data) == 0


def test_file_not_found(data_loader):
    non_existent_file = "assets/non_existent_file.txt"
    data_loader.file_paths.append(non_existent_file)
    data = data_loader.load_data()
    assert non_existent_file not in [item["file_path"] for item in data]
    assert len(data) == 3  # Original 3 files
    assert all(item["content"] is not None for item in data)
