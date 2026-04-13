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

        def _cancelled_output(msg: str) -> workflows_mistralai.ChatAssistantWorkflowOutput:
            return workflows_mistralai.ChatAssistantWorkflowOutput(
                content=[workflows_mistralai.TextOutput(text=msg)]
            )

        # ── Step 1: fetch agents + judge the current prompts (in parallel) ────
        fetch_item = workflows_mistralai.TodoListItem(
            title="Fetching agents",
            description="Retrieving current instructions from AI Studio",
        )
        judge_item = workflows_mistralai.TodoListItem(
            title="Reviewing current prompts",
            description="Judge scores strengths and weaknesses of each prompt",
        )

        async with workflows_mistralai.TodoList(items=[fetch_item, judge_item]):
            async with fetch_item:
                agent_infos: List[AgentInfo] = list(
                    await asyncio.gather(*[fetch_agent_info(aid) for aid in input.agent_ids])
                )

            async with judge_item:
                evaluations_list: List[PromptEvaluation] = list(
                    await asyncio.gather(*[
                        evaluate_prompt(info.id, info.instructions, input.judge_agent_id)
                        for info in agent_infos
                    ])
                )
        evaluations: dict[str, PromptEvaluation] = {
            info.id: ev for info, ev in zip(agent_infos, evaluations_list)
        }

        # ── Step 2: ask the reviewer how they want to start their rewrite ─────
        # Either generate an LLM draft as a starting point, or start from the
        # current prompt and write changes themselves. This avoids the wasted
        # LLM call when the reviewer already has an improved version in mind.
        n_agents = len(agent_infos)
        draft_choice = await self.wait_for_input(
            workflows_mistralai.AcceptDeclineConfirmation(
                description=(
                    f"How do you want to start your rewrite"
                    f"{'s' if n_agents > 1 else ''}?\n\n"
                    "• **Generate a draft** — an LLM produces a starting version you can edit.\n"
                    "• **Start from current** — the canvas is pre-filled with the current "
                    "prompt so you can modify it directly."
                ),
                accept_label="Generate a draft",
                decline_label="Start from current",
            )
        )
        use_llm_draft = workflows_mistralai.is_accepted(draft_choice)

        # Only call the LLM if the reviewer actually asked for a draft.
        if use_llm_draft:
            draft_item2 = workflows_mistralai.TodoListItem(
                title="Drafting starting suggestions",
                description="Generating a baseline rewrite per agent",
            )
            async with workflows_mistralai.TodoList(items=[draft_item2]):
                async with draft_item2:
                    drafts_list = list(await asyncio.gather(*[
                        generate_rewrite(info.id, info.instructions, input.rewrite_agent_id)
                        for info in agent_infos
                    ]))
            starting_prompts = dict(zip([info.id for info in agent_infos], drafts_list))
        else:
            # No LLM call — start from the current prompt as the editable baseline.
            starting_prompts = {info.id: info.instructions for info in agent_infos}

        # ── Step 3: reviewer edits the prompt for each agent ──────────────────
        final_prompts: dict[str, str] = {}

        for info in agent_infos:
            evaluation = evaluations[info.id]

            # Show the current prompt for context (read-only canvas)
            current_canvas = workflows_mistralai.CanvasResource(
                canvas=workflows_mistralai.CanvasPayload(
                    type="text/markdown",
                    title=f"Current prompt — {info.name}",
                    content=info.instructions,
                )
            )
            strengths = "\n".join(f"- {s}" for s in evaluation.strengths) or "- (none)"
            issues = "\n".join(f"- {i}" for i in evaluation.issues) or "- (none)"
            await workflows_mistralai.send_assistant_message(
                f"**{info.name}** — current prompt scored **{evaluation.score}/10** "
                f"by the judge.\n\n"
                f"**Strengths:**\n{strengths}\n\n"
                f"**Issues:**\n{issues}",
                canvas=current_canvas,
            )

            # Editable canvas pre-filled with either the LLM draft or the current
            # prompt (depending on what the reviewer chose above). They can edit
            # it, wipe it, or replace it entirely.
            edit_canvas = workflows_mistralai.CanvasResource(
                canvas=workflows_mistralai.CanvasPayload(
                    type="text/markdown",
                    title=f"Your rewrite — {info.name}",
                    content=starting_prompts[info.id],
                )
            )
            starting_from = "LLM-generated draft" if use_llm_draft else "current prompt"
            await workflows_mistralai.send_assistant_message(
                f"Edit this canvas to produce the version you want to evaluate. "
                f"It's pre-filled with the {starting_from} — tweak it or replace "
                f"it entirely with your own prompt.",
                canvas=edit_canvas,
            )
            edited = await self.wait_for_input(
                CanvasInput(
                    canvas_uri=edit_canvas.uri,
                    prompt="Submit your version when ready.",
                ),
                label="Your rewrite",
            )
            final_prompts[info.id] = edited.canvas.content.strip()

        # ── Step 4: run LLM eval against the reviewer's FINAL version ─────────
        eval_item = workflows_mistralai.TodoListItem(
            title="Running LLM eval",
            description="Judging your rewrite against the current prompt",
        )
        async with workflows_mistralai.TodoList(items=[eval_item]):
            async with eval_item:
                llm_evals = await asyncio.gather(*[
                    run_llm_eval(
                        info.id, info.name,
                        info.instructions, final_prompts[info.id],
                        info.model, input.judge_agent_id,
                    )
                    for info in agent_infos
                ])

        reviews: List[AgentReview] = [
            AgentReview(
                agent_id=info.id,
                agent_name=info.name,
                current_instructions=info.instructions,
                evaluation=evaluations[info.id],
                suggested_instructions=final_prompts[info.id],
                llm_eval=llm_eval,
            )
            for info, llm_eval in zip(agent_infos, llm_evals)
        ]

        # ── Step 5: show the report + final apply/discard decision ────────────
        report_canvas = workflows_mistralai.CanvasResource(
            canvas=workflows_mistralai.CanvasPayload(
                type="text/markdown",
                title="Evaluation Results",
                content=_fmt_report(reviews),
            )
        )
        await workflows_mistralai.send_assistant_message(
            "Here's how your rewrite scored against the current prompt:",
            canvas=report_canvas,
        )

        confirmation = await self.wait_for_input(
            workflows_mistralai.AcceptDeclineConfirmation(
                description="Apply your rewrite to the agent(s) in AI Studio?",
                accept_label="Apply changes",
                decline_label="Discard",
            )
        )

        if not workflows_mistralai.is_accepted(confirmation):
            return _cancelled_output(
                "Discarded. Your agents were not updated. The evaluation results "
                "above are still visible for reference."
            )

        # ── Step 6: apply approved changes ────────────────────────────────────
        tasks = [
            apply_prompt_update(r.agent_id, r.suggested_instructions)
            for r in reviews
        ]
        if input.emit_dataset:
            tasks += [
                emit_training_records(
                    r.agent_id, r.agent_name,
                    r.suggested_instructions,
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
