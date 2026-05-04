def route_condition(screening_scores: dict):
    anxiety = screening_scores.get("anxiety", 0)
    stress = screening_scores.get("stress", 0)
    depression = screening_scores.get("depression", 0)

    if anxiety == 0 and stress == 0 and depression == 0:
        return "neutral"

    # Route to whichever domain scored highest
    # If stress and depression both beat anxiety, route to stress
    if stress > anxiety and stress >= depression:
        return "stress"

    return "anxiety"