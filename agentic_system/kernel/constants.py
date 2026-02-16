from __future__ import annotations

CORE_NEXT_STEPS = {
    "chat_user", # chat with user for clarifying questions, status updates, or general discussion.
    "chat_subagent", # chat with sub-agents for clarifying questions, status updates, or coordination.
    "learn", # learn and understand unknowns.
    "search_web", # search the web for information.
    "retrieve_ltm", # search documents for information.
    "plan", # create a plan with multiple steps to achieve the goal.
    "act", # start the loop for doing tasks, which can include calling skills, creating sub-agents, or assigning tasks.
    "execute", # execute a specific action, which can be a skill call or a custom action.
    "verify", # verify if the current state meets the goal or if a task is completed.
    "create_sub_agent", # create a sub-agent.
    "assign_task", # assign a task to a sub-agent.
    "document", # create or update a document for knowledge storage.
    "create_skill", # create or update a reusable skill.
    "promotion_check", # check if a workflow is reusable and should be promoted to a skill.
    "report", # produce the final user-facing response.
    None, # indicates no next step and the workflow can end.
}

SUB_NEXT_STEPS = {
    "chat_subagent", # chat with core agent for clarifying questions, status updates, or general discussion.
    "learn", # learn and understand unknowns.
    "search_web", # search the web for information.
    "retrieve_ltm", # search documents for information.
    "plan", # create a plan with multiple steps to achieve the goal.
    "act", # start the loop for doing tasks.
    "execute", # execute a specific action, which can be a skill call or a custom action.
    "verify", # verify the current state meets the goal or if a task is completed with core agent.
    "document", # create or update a document for knowledge storage.
    "create_skill", # create or update a reusable skill.
    "promotion_check", # check if a workflow is reusable and should be promoted to a skill.
    "report", # produce the final response for the core agent.
    None, # indicates no next step and the workflow can end.
}

STEP_ORDER = [
    "chat_user",
    "chat_subagent",
    "learn",
    "search_web",
    "retrieve_ltm",
    "plan",
    "act",
    "execute",
    "verify",
    "create_sub_agent",
    "assign_task",
    "promotion_check",
    "create_skill",
    "document",
    "report",
    "none",
    "context",
    "do_tasks",
    "iterate",
]

TERMINAL_TOKENS = {None, "none", "no_action", "no_next_step", "null"}

DEFAULT_LIMITS = {
    "max_inner_turns": 60,
    "max_exec_actions": 40,
    "max_subagent_depth": 2,
    "token_budget": 12000,
}
