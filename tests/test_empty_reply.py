"""An empty assistant reply must never reach the saved conversation.

WHAT HAPPENED
    A user typed "hi". Nova answered with a reasoning block and no text block, so the
    visible answer was "". That blank reply was checkpointed. One minute later the same
    thread was asked "movie about kids", the whole history was replayed to Bedrock, and
    Bedrock refused it:

        ValidationException: The content field in the Message object at messages.1
        is empty. Add a ContentBlock object to the content field and try again.

    messages.1 was the blank. The request that FAILED was not the request that broke it,
    which is why nothing caught this until a trace put the two rows side by side.

WHY THE GUARD SITS IN think()
    think() is the only place a model reply enters the state. Sanitising history at send
    time would fix the symptom in one caller and leave the bad value stored.

WHY A REPLY WITH TOOL CALLS IS EXEMPT
    Its text is legitimately empty — the tool call is the content, and Bedrock encodes
    it as a toolUse block. Rewriting those would break every search the agent runs.
"""

from langchain_core.messages import AIMessage

from backend.agent import NO_ANSWER, empty_reply

REASONING_ONLY = [{"type": "reasoning_content",
                   "reasoning_content": {"text": "The user greeted me."}}]


def test_reasoning_with_no_text_is_empty():
    """The exact shape that took production down."""
    assert empty_reply(AIMessage(content=REASONING_ONLY))


def test_blank_string_is_empty():
    assert empty_reply(AIMessage(content=""))


def test_whitespace_is_empty():
    """split_content strips, so a reply of spaces is as blank as a reply of nothing."""
    assert empty_reply(AIMessage(content="   \n  "))


def test_real_answer_is_not_empty():
    assert not empty_reply(AIMessage(content="Predator is a good fit because..."))


def test_text_block_is_not_empty():
    assert not empty_reply(AIMessage(content=[{"type": "text", "text": "Alien fits."}]))


def test_tool_call_with_no_text_is_not_empty():
    """The exemption. Blank text plus a tool call is the normal, correct shape."""
    message = AIMessage(content="",
                        tool_calls=[{"name": "search_films",
                                     "args": {"query": "something tense"},
                                     "id": "call_1"}])
    assert not empty_reply(message)


def test_replacement_says_what_to_do_next():
    """A dead end that gives no way forward is only marginally better than a blank."""
    assert NO_ANSWER.strip()
    assert any(word in NO_ANSWER.lower() for word in ("mood", "genre", "film"))
