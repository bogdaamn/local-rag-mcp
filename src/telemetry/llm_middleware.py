import time

from telemetry import context, llm_cache, pricing, storage


def record_llm_call(call_site, model, prompt, temperature, fn, db_path=None):
    """Look up (prompt, model, temperature) in the response cache; on a
    miss, call fn() (a zero-arg callable returning
    (response_text, input_tokens, output_tokens)) and store the result.
    Either way, record one row in llm_calls tagged with call_site. Returns
    the response text."""
    cached = llm_cache.get(prompt, model, temperature, db_path=db_path)
    start = time.perf_counter()

    if cached is not None:
        response_text = cached["response"]
        input_tokens = cached["input_tokens"]
        output_tokens = cached["output_tokens"]
        cached_tokens = input_tokens
    else:
        response_text, input_tokens, output_tokens = fn()
        cached_tokens = 0
        llm_cache.put(
            prompt, model, temperature, response_text,
            input_tokens, output_tokens, db_path=db_path,
        )

    latency_ms = (time.perf_counter() - start) * 1000
    estimated_cost = pricing.estimate_cost(model, input_tokens, output_tokens)

    storage.insert_llm_call(
        agent_id=context.current_agent_id.get(),
        task_id=context.current_task_id.get(),
        turn_number=context.current_turn_number.get(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=0,
        latency_ms=latency_ms,
        estimated_cost=estimated_cost,
        call_site=call_site,
        db_path=db_path,
    )

    return response_text
