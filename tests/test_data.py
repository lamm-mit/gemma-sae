from dataclasses import dataclass

from gemma4_sae.data import iter_token_blocks, tokenize_document


@dataclass
class FakeTokenizer:
    eos_token_id: int = 99

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) % 20 for character in text]}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize
        assert not add_generation_prompt
        result = []
        for message in messages:
            result.extend([1 if message["role"] == "user" else 2])
            result.extend(ord(character) % 20 for character in message["content"])
        return result


def test_text_and_message_documents_use_distinct_paths() -> None:
    tokenizer = FakeTokenizer()
    text = tokenize_document(
        {"text": "abcdef"},
        tokenizer,
        column="text",
        input_format="text",
        min_chars=2,
    )
    messages = tokenize_document(
        {
            "messages": [
                {"role": "user", "content": "abc"},
                {"role": "assistant", "content": "def"},
            ]
        },
        tokenizer,
        column="messages",
        input_format="messages",
        min_chars=2,
    )
    assert text is not None and messages is not None
    assert text != messages
    assert messages[0] == 1


def test_token_blocks_pack_documents_and_insert_separator() -> None:
    tokenizer = FakeTokenizer()
    documents = [{"text": "abc"}, {"text": "def"}]
    blocks = list(
        iter_token_blocks(
            documents,
            tokenizer,
            column="text",
            input_format="text",
            sequence_length=4,
            min_chars=1,
        )
    )
    assert len(blocks) == 2
    assert 99 in blocks[0].tolist()
