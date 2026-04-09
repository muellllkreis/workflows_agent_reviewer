"""
Agent Prompt Reviewer
=====================
A meta workflow: uses Mistral Workflows to review and improve Mistral Agents.

For each agent, it runs in parallel:
  ✦ Prompt quality evaluation   — LLM-as-judge scores the current instructions
  ✦ Prompt rewrite              — LLM generates an improved version

Then, using both results, it runs:
  ✦ LLM evaluation              — judge scores before vs. after across 3 generated
                                   test prompts, producing a concrete score delta

Then it durably pauses and waits for a human reviewer to approve or reject.
If approved, it patches each agent's instructions via the Mistral API.

Features showcased
------------------
  asyncio.gather     — parallel eval + rewrite per agent, all agents concurrent
  @workflow.signal   — reviewer sends approve/reject at any time (human-in-the-loop)
  @workflow.query    — inspect stage + scores without interrupting execution
  wait_condition     — durable pause that survives worker restarts
  Real API calls     — GET /v1/agents/{id}, chat completions, PATCH /v1/agents/{id}

Quick start
-----------
  # Review a single agent (default):
  python src/workflows/start.py --workflow agent-prompt-reviewer --input '{}'

  # Review two agents side by side:
  python src/workflows/start.py --workflow agent-prompt-reviewer --input '{
    "agent_ids": ["ag_019cbcd3417c749eb7420e7b78c13967", "<second_agent_id>"]
  }'

  # While the workflow is paused, poll its status:
  #   → use the AI Studio UI or the WorkflowsClient query API with name "review-status"

  # Approve the changes:
  #   → send signal "reviewer-decision" with {"approved": true, "comment": "Ship it."}

  # Reject the changes:
  #   → send signal "reviewer-decision" with {"approved": false, "comment": "Needs more work."}
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import List

from pydantic import BaseModel

import mistralai.workflows as workflows
from mistralai.workflows.client import get_mistral_client


# ── Models ────────────────────────────────────────────────────────────────────

class AgentReviewInput(BaseModel):
    agent_ids: List[str] = ["ag_019cbcd3417c749eb7420e7b78c13967"]
    # Optional human-written rewrites keyed by agent_id.
    # When provided for an agent, the rewrite agent is skipped entirely — the human
    # version is used directly for evaluation and (if approved) applied to the agent.
    # If omitted for an agent, falls back to the LLM rewrite agent.
    rewrites: dict[str, str] = {}
    # Agents used for the rewrite (fallback) and judging steps
    rewrite_agent_id: str = "ag_019d5e82a5d4771ab4e3849ddc498084"
    judge_agent_id: str = "ag_019d5e8348d577df92e43328385ce800"
    # Set to true to emit winning (test_prompt, better_response) pairs into a dataset after approval
    # Requires an AI Studio enterprise plan
    emit_dataset: bool = False
    dataset_name: str = "agent-prompt-reviewer"


class AgentInfo(BaseModel):
    id: str
    name: str
    instructions: str
    model: str  # the model powering the agent (used for before/after response generation)


class PromptEvaluation(BaseModel):
    score: int           # 1–10 overall quality
    strengths: List[str]
    issues: List[str]


class LLMEvalResult(BaseModel):
    test_prompts: List[str]        # auto-generated test messages
    after_responses: List[str]     # responses under the suggested instructions (for dataset export)
    before_scores: List[float]     # per-prompt judge scores under current instructions
    after_scores: List[float]      # per-prompt judge scores under suggested instructions
    before_avg_score: float        # avg judge score under current instructions
    after_avg_score: float         # avg judge score under suggested instructions
    score_delta: float             # positive = improvement
    judge_notes: str


class AgentReview(BaseModel):
    agent_id: str
    agent_name: str
    current_instructions: str
    evaluation: PromptEvaluation
    suggested_instructions: str
    llm_eval: LLMEvalResult


class ReviewerDecision(BaseModel):
    approved: bool
    comment: str = ""


class ReviewStatus(BaseModel):
    stage: str            # "fetching" | "analyzing" | "awaiting_approval" | "applying" | "done"
    agents_reviewed: int
    total_agents: int
    avg_score_delta: float
    decision: str         # "pending" | "approved" | "rejected"


class ReviewReport(BaseModel):
    agent_reviews: List[AgentReview]
    avg_score_delta: float
    approved: bool
    reviewer_comment: str
    updates_applied: bool
    dataset_id: str | None = None   # populated when training records were emitted


# ── Shared helper ─────────────────────────────────────────────────────────────

def _client():
    return get_mistral_client(api_key=os.environ["MISTRAL_API_KEY"])


# ── Activities ────────────────────────────────────────────────────────────────

@workflows.activity(name="fetch-agent-info")
async def fetch_agent_info(agent_id: str) -> AgentInfo:
    """Retrieve agent metadata from Mistral AI Studio."""
    client = _client()
    agent = await client.beta.agents.get_async(agent_id=agent_id)
    return AgentInfo(
        id=agent.id,
        name=agent.name or agent_id,
        instructions=agent.instructions or "(no instructions set)",
        model=agent.model or "mistral-small-latest",
    )


@workflows.activity(name="evaluate-prompt")
async def evaluate_prompt(
    agent_id: str,
    instructions: str,
    judge_agent_id: str,
) -> PromptEvaluation:
    """
    LLM-as-judge: score the current instructions and list strengths / issues.
    Uses the dedicated judge agent defined in AI Studio.
    """
    client = _client()
    response = await client.agents.complete_async(
        agent_id=judge_agent_id,
        messages=[
            {
                "role": "user",
                "content": f"Evaluate this system prompt:\n\n<prompt>\n{instructions}\n</prompt>",
            },
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return PromptEvaluation(
        score=int(data.get("score", 5)),
        strengths=data.get("strengths", []),
        issues=data.get("issues", []),
    )


@workflows.activity(name="generate-rewrite")
async def generate_rewrite(
    agent_id: str,
    instructions: str,
    rewrite_agent_id: str,
) -> str:
    """
    LLM rewrite: produce an improved version of the instructions.
    Uses the dedicated rewrite agent defined in AI Studio.
    Runs in parallel with evaluate_prompt — independently identifies and fixes issues.
    """
    client = _client()
    response = await client.agents.complete_async(
        agent_id=rewrite_agent_id,
        messages=[
            {
                "role": "user",
                "content": f"Rewrite this system prompt:\n\n<prompt>\n{instructions}\n</prompt>",
            },
        ],
    )
    return response.choices[0].message.content.strip()


@workflows.activity(name="run-llm-eval")
async def run_llm_eval(
    agent_id: str,
    agent_name: str,
    current_instructions: str,
    suggested_instructions: str,
    agent_model: str,
    judge_agent_id: str,
) -> LLMEvalResult:
    """
    LLM-as-judge evaluation comparing the current vs. suggested prompt:

      1. Generate 3 test prompts tailored to the agent's purpose (via judge agent).
      2. Get responses under both the current and suggested instructions.
         (Both sets run concurrently via asyncio.gather.)
      3. The judge agent scores each response pair 1–10 on helpfulness and task-adherence.

    Returns before/after average scores and the delta (positive = improvement).
    """
    client = _client()

    # Step 1 — generate test prompts relevant to this agent
    tp_resp = await client.agents.complete_async(
        agent_id=judge_agent_id,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Given this system prompt for an AI assistant called '{agent_name}':\n\n"
                    f"<prompt>\n{current_instructions}\n</prompt>\n\n"
                    "Generate exactly 3 short, realistic user messages to test this agent. "
                    'Return ONLY a JSON array of 3 strings, e.g. ["msg1", "msg2", "msg3"].'
                ),
            }
        ],
        response_format={"type": "json_object"},
    )
    raw = json.loads(tp_resp.choices[0].message.content)
    # Model may return {"prompts": [...]} or a bare list — handle both
    test_prompts: List[str] = raw if isinstance(raw, list) else next(iter(raw.values()))
    test_prompts = test_prompts[:3]

    # Steps 2 & 3 — for each test prompt, get before/after responses and judge them
    before_scores: List[float] = []
    after_scores: List[float] = []
    after_responses: List[str] = []
    judge_notes_parts: List[str] = []

    for prompt in test_prompts:
        # Get both responses concurrently
        before_resp, after_resp = await asyncio.gather(
            client.chat.complete_async(
                model=agent_model,
                messages=[
                    {"role": "system", "content": current_instructions},
                    {"role": "user", "content": prompt},
                ],
            ),
            client.chat.complete_async(
                model=agent_model,
                messages=[
                    {"role": "system", "content": suggested_instructions},
                    {"role": "user", "content": prompt},
                ],
            ),
        )
        before_text = before_resp.choices[0].message.content
        after_text = after_resp.choices[0].message.content
        after_responses.append(after_text)

        # Judge scores both
        judge_resp = await client.agents.complete_async(
            agent_id=judge_agent_id,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"User prompt: {prompt}\n\n"
                        f"Response A (current prompt):\n{before_text}\n\n"
                        f"Response B (suggested prompt):\n{after_text}\n\n"
                        "Score each response 1–10 on helpfulness and task-adherence. "
                        'Return ONLY JSON: {"score_a": <int>, "score_b": <int>, "note": "<str>"}'
                    ),
                }
            ],
            response_format={"type": "json_object"},
        )
        scores = json.loads(judge_resp.choices[0].message.content)
        before_scores.append(float(scores.get("score_a", 5)))
        after_scores.append(float(scores.get("score_b", 5)))
        judge_notes_parts.append(scores.get("note", ""))

    before_avg = round(sum(before_scores) / len(before_scores), 2)
    after_avg = round(sum(after_scores) / len(after_scores), 2)

    return LLMEvalResult(
        test_prompts=test_prompts,
        after_responses=after_responses,
        before_scores=before_scores,
        after_scores=after_scores,
        before_avg_score=before_avg,
        after_avg_score=after_avg,
        score_delta=round(after_avg - before_avg, 2),
        judge_notes=" | ".join(judge_notes_parts),
    )


@workflows.activity(name="emit-training-records")
async def emit_training_records(
    agent_id: str,
    agent_name: str,
    suggested_instructions: str,
    test_prompts: List[str],
    after_responses: List[str],
    before_scores: List[float],
    after_scores: List[float],
    dataset_name: str,
) -> str:
    """
    Write winning (prompt, response) pairs to an AI Studio dataset.

    Only emits records where the suggested instructions produced a strictly better
    response (after_score > before_score), so the dataset only contains examples
    where the rewrite made a measurable difference — useful for fine-tuning or
    as a reference eval set.

    Returns the dataset ID.
    """
    from mistralai.client.models.conversationpayload import ConversationPayload

    client = _client()

    # Create a new dataset for this run (name includes agent for traceability)
    dataset = await client.beta.observability.datasets.create_async(
        name=f"{dataset_name} — {agent_name}",
        description=(
            f"Winning prompt/response pairs from Agent Prompt Reviewer. "
            f"Agent: {agent_id}. Generated by automated LLM-as-judge eval."
        ),
    )

    # Emit one record per test prompt that improved under the new instructions
    for prompt, response, before, after in zip(
        test_prompts, after_responses, before_scores, after_scores
    ):
        if after <= before:
            continue  # skip prompts where the rewrite didn't help

        await client.beta.observability.datasets.create_record_async(
            dataset_id=dataset.id,
            payload=ConversationPayload(
                messages=[
                    {"role": "system", "content": suggested_instructions},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            ),
            properties={
                "agent_id": agent_id,
                "before_score": before,
                "after_score": after,
                "score_delta": round(after - before, 2),
            },
        )

    return dataset.id


@workflows.activity(name="apply-prompt-update")
async def apply_prompt_update(agent_id: str, new_instructions: str) -> str:
    """PATCH the agent's instructions in Mistral AI Studio."""
    client = _client()
    updated = await client.beta.agents.update_async(
        agent_id=agent_id,
        instructions=new_instructions,
        version_message="Prompt updated by Agent Prompt Reviewer workflow",
    )
    return updated.id


# ── Workflow ──────────────────────────────────────────────────────────────────

@workflows.workflow.define(
    name="agent-prompt-reviewer",
    workflow_display_name="Agent Prompt Reviewer",
    workflow_description=(
        "Reviews and improves Mistral agent system prompts using LLM-as-judge evaluation, "
        "then gates on human approval before applying changes in AI Studio."
    ),
)
class AgentPromptReviewer:
    def __init__(self) -> None:
        self._stage = "starting"
        self._reviews: List[AgentReview] = []
        self._total: int = 0
        self._decision: ReviewerDecision | None = None

    # ── Signal: reviewer posts their decision ─────────────────────────────────

    @workflows.workflow.signal(
        name="reviewer-decision",
        description="Approve or reject the proposed prompt updates.",
    )
    async def reviewer_decision(self, decision: ReviewerDecision) -> None:
        """
        Send this after reviewing the report to unblock the workflow.
        The workflow is durably paused at wait_condition — even if the worker
        restarts while waiting, execution resumes from this exact point.
        """
        self._decision = decision

    # ── Query: inspect progress at any time ───────────────────────────────────

    @workflows.workflow.query(
        name="review-status",
        description="Return the current stage, per-agent progress, and avg score delta.",
    )
    def review_status(self) -> ReviewStatus:
        avg_delta = (
            round(sum(r.llm_eval.score_delta for r in self._reviews) / len(self._reviews), 2)
            if self._reviews
            else 0.0
        )
        if self._decision is None:
            decision_str = "pending"
        elif self._decision.approved:
            decision_str = "approved"
        else:
            decision_str = "rejected"

        return ReviewStatus(
            stage=self._stage,
            agents_reviewed=len(self._reviews),
            total_agents=self._total,
            avg_score_delta=avg_delta,
            decision=decision_str,
        )

    # ── Entrypoint ────────────────────────────────────────────────────────────

    @workflows.workflow.entrypoint
    async def run(self, input: AgentReviewInput) -> ReviewReport:
        self._total = len(input.agent_ids)

        # Step 1 — fetch all agents concurrently
        self._stage = "fetching"
        agent_infos: List[AgentInfo] = list(
            await asyncio.gather(*[fetch_agent_info(aid) for aid in input.agent_ids])
        )

        # Step 2 — analyse every agent concurrently.
        #   Per agent:  Phase A runs evaluate + rewrite in parallel (2 LLM calls at once)
        #               Phase B runs the LLM eval once both results are available
        self._stage = "analyzing"

        async def _review_one(info: AgentInfo) -> AgentReview:
            # Phase A — quality evaluation always runs.
            # Rewrite uses the human-provided version if supplied, otherwise falls back
            # to the rewrite agent. Both paths run in parallel when using the LLM fallback.
            human_rewrite = input.rewrites.get(info.id)
            if human_rewrite:
                evaluation = await evaluate_prompt(info.id, info.instructions, input.judge_agent_id)
                suggested = human_rewrite
            else:
                evaluation, suggested = await asyncio.gather(
                    evaluate_prompt(info.id, info.instructions, input.judge_agent_id),
                    generate_rewrite(info.id, info.instructions, input.rewrite_agent_id),
                )
            # Phase B — sequential: LLM eval needs the rewrite to compare against
            llm_eval = await run_llm_eval(
                info.id,
                info.name,
                info.instructions,
                suggested,
                info.model,
                input.judge_agent_id,
            )
            return AgentReview(
                agent_id=info.id,
                agent_name=info.name,
                current_instructions=info.instructions,
                evaluation=evaluation,
                suggested_instructions=suggested,
                llm_eval=llm_eval,
            )

        # All agents are reviewed concurrently
        self._reviews = list(
            await asyncio.gather(*[_review_one(info) for info in agent_infos])
        )

        # Step 3 — durably block until the reviewer sends their decision.
        #   The workflow checkpoints here. A worker restart won't lose any state.
        self._stage = "awaiting_approval"
        await workflows.workflow.wait_condition(
            lambda: self._decision is not None,
        )

        # Step 4 — apply approved changes and emit training data concurrently
        updates_applied = False
        dataset_ids: List[str] = []
        if self._decision.approved:
            self._stage = "applying"
            tasks = [apply_prompt_update(r.agent_id, r.suggested_instructions) for r in self._reviews]
            if input.emit_dataset:
                tasks += [
                    emit_training_records(
                        r.agent_id,
                        r.agent_name,
                        r.suggested_instructions,
                        r.llm_eval.test_prompts,
                        r.llm_eval.after_responses,
                        r.llm_eval.before_scores,
                        r.llm_eval.after_scores,
                        input.dataset_name,
                    )
                    for r in self._reviews
                ]
            results = await asyncio.gather(*tasks)
            updates_applied = True
            if input.emit_dataset:
                dataset_ids = list(results[len(self._reviews):])

        self._stage = "done"

        avg_delta = (
            round(sum(r.llm_eval.score_delta for r in self._reviews) / len(self._reviews), 2)
            if self._reviews
            else 0.0
        )

        return ReviewReport(
            agent_reviews=self._reviews,
            avg_score_delta=avg_delta,
            approved=self._decision.approved,
            reviewer_comment=self._decision.comment,
            updates_applied=updates_applied,
            dataset_id=dataset_ids[0] if dataset_ids else None,
        )
