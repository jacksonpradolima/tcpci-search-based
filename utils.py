import random


def sort_update_actions(individual, actions):
    pos = individual
    i = 0

    for action in actions:
        action['CalcPrio'] = pos[i]
        i += 1

    # Sort tc by Prio ASC (for backwards scheduling), break ties randomly
    return sorted(actions, key=lambda x: (x['CalcPrio'], random.random()))
