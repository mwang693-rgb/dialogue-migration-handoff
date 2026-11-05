"""Controlled scripts, with no recycled assistant text or response-dependent users.

These are deliberately small fixtures. Variations within a family are NOT evidence
of transfer to unseen task families; the pilot is only a feasibility diagnostic.
"""
import json


def make_episodes(count=30, dev_count=20):
    rows = []
    for i in range(count):
        family = ["workflow", "missing_field", "confirmed_state"][i % 3]
        # Distinct script instances, but coarse families are shared across splits.
        item = ["conference", "workshop", "seminar", "meetup", "retreat"][i % 5]
        budget = 100 + 13 * i
        date = f"2027-03-{1 + i % 20:02d}"
        new_date = f"2027-04-{1 + i % 20:02d}"
        city = ["Madison", "Chicago", "Seattle", "Boston", "Austin"][i % 5]
        venue = f"Hall-{i + 1}"
        suggestion = f"Garden-{i + 1}"
        if family == "workflow":
            steps = ["collect_requirements", "compare_options", "finalize_plan"]
            users = [
                f"For planning this {item}, use exactly this three-step workflow: "
                f"1 {steps[0]}, 2 {steps[1]}, 3 {steps[2]}. Only do one step per reply. "
                "Keep track of completed steps rather than restarting each time.",
                "Step 1 is now complete. We have all the requirements. "
                "Step 2 has not been completed.",
                "Keep step 2 as the next pending step. Do not execute it yet. "
                "Wait until I explicitly tell you to continue.",
            ]
            specs = [
                ("continue", "Continue.", steps, 1, {"step": steps[1]}),
                ("new_task", f"Start a separate new {item} using the same workflow. "
                 "No steps have been completed for this new task.", steps, 0,
                 {"step": steps[0]}),
                ("update", "For this task, skip the pending comparison step. "
                 "Go directly to finalizing the plan.", steps, 2,
                 {"step": steps[2]}),
            ]
            schema = 'Return only a JSON object with one key "step" containing the next step name.'
        elif family == "missing_field":
            users = [
                f"Help me plan a {item}. Before recommending anything, you must know "
                "the city, date, and budget. If a required field is missing, ask for it. "
                "Do not invent values for missing fields.",
                f"The city is {city}, and the date is {date}. I have not specified the budget.",
                "The budget is still unknown. Keep the city and date I gave you. "
                "We still require all three fields before making recommendations.",
            ]
            options = ["ask_for_budget", "proceed_with_known_fields", "ask_for_city"]
            specs = [
                ("continue", "Go ahead with the planning now.", options, 0,
                 {"action": "ask_for_budget", "city": city, "date": date, "budget": None}),
                ("new_info", f"The budget is {budget}. Proceed with the planning.", options, 1,
                 {"action": "proceed_with_known_fields", "city": city, "date": date, "budget": budget}),
                ("update", "For this task, remove budget from the required fields. "
                 "Proceed using the known city and date; leave the unknown budget unset.", options, 1,
                 {"action": "proceed_with_known_fields", "city": city, "date": date, "budget": None}),
            ]
            schema = ('Return only a JSON object with keys "action", "city", "date", "budget". '
                      'Use action "ask_for_budget", "proceed_with_known_fields", or "ask_for_city" '
                      'according to what you should do. Use null for an unknown budget.')
        else:
            users = [
                f"Plan a {item} for {date} with budget {budget}. The venue is undecided. "
                "A suggestion is not a confirmed choice unless I explicitly approve it.",
                f"Consider {suggestion} as a possible venue, but do not treat it as confirmed.",
                f"I now confirm venue {venue}, and change the date to {new_date}. "
                "Keep the original budget. Do not book yet.",
            ]
            options = [f"use_confirmed_venue_{venue}", f"use_tentative_venue_{suggestion}", "ask_for_venue"]
            specs = [
                ("continue", "Proceed with the booking.", options, 0,
                 {"action": "book", "venue": venue, "date": new_date, "budget": budget}),
                ("new_info", f"Change the budget to {budget + 50} and proceed with booking. "
                 "Keep the confirmed date and venue.", options, 0,
                 {"action": "book", "venue": venue, "date": new_date, "budget": budget + 50}),
                ("update", "Cancel my venue confirmation. Neither venue is confirmed now. "
                 "Keep the latest date and original budget, and help me proceed.", options, 2,
                 {"action": "ask_for_venue", "venue": None, "date": new_date, "budget": budget}),
            ]
            schema = ('Return only a JSON object with keys "action", "venue", "date", "budget". '
                      'Use action "book" if a venue is confirmed, otherwise "ask_for_venue". '
                      'Use null for an unconfirmed venue.')
        queries = []
        for qid, text, actions, gold, expected in specs:
            queries.append(dict(query_id=qid, text=text, actions=actions,
                                gold_action=gold, free_query=text + "\n" + schema,
                                expected_json=expected))
        rows.append(dict(episode_id=f"pilot-{i:03d}", family=family,
                         source_group=f"script-{i:03d}", split="dev" if i < dev_count else "validation",
                         users=users, queries=queries))
    return rows


def check_json(text, expected):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        actual = json.loads(cleaned)
    except (ValueError, TypeError):
        return {"valid_json": False, "all_fields": False, "field_pass": {k: False for k in expected}}
    if not isinstance(actual, dict):
        return {"valid_json": False, "all_fields": False, "field_pass": {k: False for k in expected}}
    passes = {k: k in actual and actual[k] == v for k, v in expected.items()}
    return {"valid_json": True, "all_fields": all(passes.values()), "field_pass": passes}


def self_check():
    episodes = make_episodes()
    assert len(episodes) == 30
    assert sum(e["split"] == "dev" for e in episodes) == 20
    assert len({e["source_group"] for e in episodes}) == 30
    for e in episodes:
        assert len(e["users"]) == 3 and len(e["queries"]) == 3
        for q in e["queries"]:
            assert len(q["actions"]) == 3 and 0 <= q["gold_action"] < 3
            assert check_json(json.dumps(q["expected_json"]), q["expected_json"])["all_fields"]
            assert not check_json("not JSON", q["expected_json"])["all_fields"]
            changed = dict(q["expected_json"])
            changed[next(iter(changed))] = "incorrect-value"
            assert not check_json(json.dumps(changed), q["expected_json"])["all_fields"]
    print("30 script fixtures and success/failure checkers verified")


if __name__ == "__main__":
    self_check()
