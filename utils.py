import matplotlib.pyplot as plt
from scipy.stats import f_oneway, sem, ttest_ind
import numpy as np



a = 1
selfEfficacy = 1
costCoefficient = .185  # .1 low cost; .8 med cost; 2.0 high cost
tau = .7

def encoder(l):
    encodedList = []
    for i in range(len(l)):
        temp = [0 for _ in l]
        temp[i] = 1
        encodedList.append(temp)

    return encodedList


def getResults(id, groupSizes, stateAndDriveToSatisfaction, states, driveAndStateToRelevance, stateToStimulusMapping, deficits, rules, goals, effort):
    res = np.zeros(shape=(len(states), len(rules)))
    n = groupSizes[id]
    for i in range(n):
        r = getboltzmannDistOfUtility(rules=rules, states=states,
                                             i=i,
                                             stateAndDriveToSatisfaction=stateAndDriveToSatisfaction,
                                             driveAndStateToRelevance=driveAndStateToRelevance,
                                             stateToStimulusMapping=stateToStimulusMapping, deficits=deficits,
                                             goals=goals, effort=effort)
        res += np.array(r)

    return res


def calcPerformance(res, mult):

    n = int(sum(res[0]))
    # print(res)

    score = []
    all_scores = []
    for r in res:
        aa = 0
        scoresUnderCondition = []
        for i in range(len(r)):
            aa += r[i] * (i + 1)
            for j in range(int(r[i])):
                scoresUnderCondition.append((i + 1)*mult)
        score.append(aa*mult / n)
        all_scores.append(scoresUnderCondition)

    sd_result = []
    se_result = []
    for s in all_scores:
        sd_result.append(np.std(s))
        se_result.append(sem(s))

    return score, sd_result, se_result

def resToFreq(res):
    # print(res)
    n = int(sum(res[0]))
    nStates = int(len(res))
    results = np.zeros(shape=(nStates, n))

    for i in range(nStates):
        start = 0
        state = res[i]
        for j in range(len(state)):
            nDecisions = int(state[j])
            results[i, start:start + nDecisions] = np.ones(shape=(nDecisions)) * j
            start += nDecisions
    # print(results)
    return results

def anova(res):
    df1 = res.shape[0]-1
    df2 = res[0].shape[0]*res.shape[0]

    anova_res = f_oneway(*res)
    f, p = anova_res[0], anova_res[1]

    out = 'F('+str(df1)+','+str(df2)+') = '+str(f)+' p = ' + str(p)

    return out


def normalDist(l, n, sigma=.1):
    for i in range(len(l)):
        mu = l[i] # mean
        l[i] = list(np.random.normal(mu, sigma, n))

    return l


def boltzmann(U, ruleIndex, t):

    s = np.sum([np.exp(util/t) for util in U])

    return np.exp(U[ruleIndex]/t)/s


coefA = 1


def getboltzmannDistOfUtility(rules, states, i, stateAndDriveToSatisfaction, driveAndStateToRelevance, stateToStimulusMapping, deficits, goals, effort):

    utility = []
    totalRes = []
    print('ls',len(states))
    for j in range(len(states)):
        print('\nj', j)
        res = [0 for i in range(len(rules))]
        stimulusActivation = np.dot(states[j], stateToStimulusMapping)
        # Step 1. Calculate Drive Strength
        driveStrength = np.multiply(stimulusActivation, [a[i] for a in deficits])
        # print(driveStrength[1], '\t', np.argmax(states[j]), '\tHigh')#, stimulusActivation, [a[i] for a in deficits])

        # Step 2. Calculate Goal Strength
        goalIndex = 0

        if driveAndStateToRelevance:
            goalStrength = []
            for d in range(len(goals)):
                print(driveAndStateToRelevance[d][j], driveStrength)
                goalStrength.append(np.sum(np.dot(driveAndStateToRelevance[d][j], driveStrength)))
            print(goalStrength)
            boltzmannDistOfGoals = []
            for e in range(len(goals)):
                boltzmannDistOfGoals.append(boltzmann(goalStrength, e, tau))
            print(boltzmannDistOfGoals)
            goal = np.random.choice(goals, p=boltzmannDistOfGoals)
            print(goal)
            goalIndex = goals.index(goal)

        # Step 3 Calculate Utility
        # Calculate Utility for each rule
        ruleUtility = []
        for rule in rules:
            u = effort[rule]*(coefA * selfEfficacy * np.dot(driveStrength, np.array(stateAndDriveToSatisfaction[goalIndex][j]).T) - costCoefficient)
            # print(u)
            ruleUtility.append(u)
        # Add 'Utility for each rule' list to the list of all utilities
        utility.append(ruleUtility)
        # Step 4. Calculate P(Rule | State) = Boltzmann Distribution of Utility
        boltzmannDistUtilityPerState = []
        # print(utility[j])
        for k in range(len(rules)):
            boltzmannDistUtilityPerState.append(boltzmann(utility[j], k, tau))

        action = np.random.choice(rules, p=boltzmannDistUtilityPerState)
        k = rules.index(action)
        res[k] += 1

        totalRes.append(res)
    print('tr',totalRes)
    return totalRes


def plot_results(groups, scores, states, plot_name):
    # Final step. Plotting the results (Utility)
    colors = ['black', 'grey', 'blue', 'cyan', 'magenta']
    s = scores
    # max_score = []
    #
    # for score in scores:
    #     max_score.append(np.max(score))

    n_groups = len(states)

    # create plot
    plt.subplots()
    index = list(range(n_groups))
    bar_width = 0.1
    opacity = 1
    for l in range(len(groups)):
        plt.bar([i + bar_width*l for i in index],
                s[l], bar_width, alpha=opacity,
                color=colors[l],
                label=groups[l])

    plt.xlabel(' ')
    plt.ylabel('Score')
    plt.title('Performance (simulated)')
    plt.xticks([i + bar_width for i in index], states)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_name+'_simulation.png')

def t_test(states, res, df=None):
    out =''
    deg_free = 0
    if df:
        deg_free = df
    else:
        for r in res:
            deg_free += sum(r)

    for i in range(len(states)):
        for j in range(len(states)):
            if i < j:
                ttest = ttest_ind(resToFreq(res)[i], resToFreq(res)[j])
                out += str(states[i]) + ' vs ' + str(states[j]) + ' t-test: t(' + str(int(deg_free)) + ') = ' + str(ttest[0]) + ' p = ' + str(ttest[1]) + str('\n')

    return out
