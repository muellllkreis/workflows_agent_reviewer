"""
Agent Prompt Reviewer — Interactive (Le Chat) version
======================================================
Same logic as agent_prompt_reviewer.py, but surfaced as a Le Chat conversation
using InteractiveWorkflow.  No terminal commands, no execution IDs, no JSON payloads.

The reviewer's experience:
  1. Triggers the workflow from Le Chat (or it's published as an assistant).
  2. Watches a live todo-list while the analysis runs.
  3. Reads the evaluation report rendered as a markdown canvas.
  4. Edits the suggested prompt inline in the canvas — or leaves it as-is.
  5. Clicks Approve or Reject.
  6. Done — the agent is updated (or not) and Le Chat shows the outcome.

This replaces:
  - The `rewrites` input dict  → canvas edit IS the human rewrite step
  - The `wait_condition` + signal → `wait_for_input` with AcceptDeclineConfirmation
  - The `make query` / `make signal` terminal dance → native Le Chat UI
"""

from __future__ import annotations

import asyncio
import os
from typing import List

from pydantic import BaseModel

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from mistralai.workflows.conversational import CanvasInput

from mistralai.workflows.client import get_mistral_client

# Re-use all models and activities from the non-interactive version
from workflows.agent_prompt_reviewer import (
    AgentInfo,
    AgentReview,
    AgentReviewInput,
    LLMEvalResult,
    PromptEvaluation,
    apply_prompt_update,
    emit_training_records,
    evaluate_prompt,
    fetch_agent_info,
    generate_rewrite,
    run_llm_eval,
)


def _fmt_report(reviews: List[AgentReview]) -> str:
    """Render the review results as markdown for the canvas."""
    lines = ["# Agent Prompt Review\n"]
    for r in reviews:
        delta_sign = "+" if r.llm_eval.score_delta >= 0 else ""
        lines += [
            f"## {r.agent_name}",
            f"**Current prompt score:** {r.evaluation.score}/10",
            "",
            "**Strengths:**",
            *[f"- {s}" for s in r.evaluation.strengths],
            "",
            "**Issues found:**",
            *[f"- {i}" for i in r.evaluation.issues],
            "",
            f"**LLM eval score delta:** {delta_sign}{r.llm_eval.score_delta} "
            f"({r.llm_eval.before_avg_score} → {r.llm_eval.after_avg_score})",
            f"*{r.llm_eval.judge_notes}*",
            "",
        ]
    return "\n".join(lines)


@workflows.workflow.define(
    name="agent-prompt-reviewer-interactive",
    workflow_display_name="Agent Prompt Reviewer",
    workflow_description=(
        "Reviews your agent's system prompt, suggests improvements, "
        "and lets you edit and approve changes — all from this conversation."
    ),
)
class AgentPromptReviewerInteractive(workflows.InteractiveWorkflow):

    @workflows.workflow.entrypoint
    async def run(self, input: AgentReviewInput) -> workflows_mistralai.ChatAssistantWorkflowOutput:

        # ── Step 1: fetch + analyse with a live progress list ─────────────────
        fetch_item = workflows_mistralai.TodoListItem(
            title="Fetching agents",
            description="Retrieving current instructions from AI Studio",
        )
        analyse_item = workflows_mistralai.TodoListItem(
            title="Analysing prompts",
            description="Running parallel evaluation and rewrite",
        )
        eval_item = workflows_mistralai.TodoListItem(
            title="Running LLM eval",
            description="Comparing before/after with judge",
        )

        reviews: List[AgentReview] = []

        async with workflows_mistralai.TodoList(
            items=[fetch_item, analyse_item, eval_item]
        ):
            async with fetch_item:
                agent_infos: List[AgentInfo] = list(
                    await asyncio.gather(*[fetch_agent_info(aid) for aid in input.agent_ids])
                )

            async with analyse_item:
                evals_and_rewrites = await asyncio.gather(*[
                    asyncio.gather(
                        evaluate_prompt(info.id, info.instructions, input.judge_agent_id),
                        generate_rewrite(info.id, info.instructions, input.rewrite_agent_id),
                    )
                    for info in agent_infos
                ])

            async with eval_item:
                llm_evals = await asyncio.gather(*[
                    run_llm_eval(
                        info.id, info.name,
                        info.instructions, suggested,
                        info.model, input.judge_agent_id,
                    )
                    for info, (_, suggested) in zip(agent_infos, evals_and_rewrites)
                ])

            for info, (evaluation, suggested), llm_eval in zip(
                agent_infos, evals_and_rewrites, llm_evals
            ):
                reviews.append(AgentReview(
                    agent_id=info.id,
                    agent_name=info.name,
                    current_instructions=info.instructions,
                    evaluation=evaluation,
                    suggested_instructions=suggested,
                    llm_eval=llm_eval,
                ))

        # ── Step 2: show the report + let the reviewer edit each prompt ────────
        report_canvas = workflows_mistralai.CanvasResource(
            canvas=workflows_mistralai.CanvasPayload(
                type="text/markdown",
                title="Prompt Review Report",
                content=_fmt_report(reviews),
            )
        )
        await workflows_mistralai.send_assistant_message(
            "Analysis complete. Here's the report:",
            canvas=report_canvas,
        )

        # One canvas per agent — reviewer can edit the suggested prompt directly
        final_prompts: dict[str, str] = {}
        for review in reviews:
            prompt_canvas = workflows_mistralai.CanvasResource(
                canvas=workflows_mistralai.CanvasPayload(
                    type="text/markdown",
                    title=f"Suggested prompt — {review.agent_name}",
                    content=review.suggested_instructions,
                )
            )
            await workflows_mistralai.send_assistant_message(
                f"Here's the suggested new prompt for **{review.agent_name}**. "
                "Feel free to edit it before approving.",
                canvas=prompt_canvas,
            )
            edited = await self.wait_for_input(
                CanvasInput(
                    canvas_uri=prompt_canvas.uri,
                    prompt="Any changes? (edit above or leave as-is)",
                ),
                label="Edit prompt",
            )
            final_prompts[review.agent_id] = edited.canvas.content.strip()

        # ── Step 3: final approval ─────────────────────────────────────────────
        confirmation = await self.wait_for_input(
            workflows_mistralai.AcceptDeclineConfirmation(
                description="Apply these prompt updates to your agents in AI Studio?",
                accept_label="Yes, apply",
                decline_label="No, discard",
            )
        )

        if not workflows_mistralai.is_accepted(confirmation):
            return workflows_mistralai.ChatAssistantWorkflowOutput(
                content=[workflows_mistralai.TextOutput(
                    text="No changes made. Your agents are unchanged."
                )]
            )

        # ── Step 4: apply ──────────────────────────────────────────────────────
        tasks = [
            apply_prompt_update(r.agent_id, final_prompts[r.agent_id])
            for r in reviews
        ]
        if input.emit_dataset:
            tasks += [
                emit_training_records(
                    r.agent_id, r.agent_name,
                    final_prompts[r.agent_id],
                    r.llm_eval.test_prompts,
                    r.llm_eval.after_responses,
                    r.llm_eval.before_scores,
                    r.llm_eval.after_scores,
                    input.dataset_name,
                )
                for r in reviews
            ]
        await asyncio.gather(*tasks)

        avg_delta = round(
            sum(r.llm_eval.score_delta for r in reviews) / len(reviews), 2
        )
        delta_sign = "+" if avg_delta >= 0 else ""

        return workflows_mistralai.ChatAssistantWorkflowOutput(
            content=[workflows_mistralai.TextOutput(
                text=(
                    f"Done! Updated {len(reviews)} agent(s). "
                    f"Average eval score improvement: {delta_sign}{avg_delta}."
                )
            )]
        )
