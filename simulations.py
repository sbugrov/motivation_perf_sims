# from cu import *
import clarion_utils as cu
from random import randint
import scipy
import numpy as np

'''
The ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid.

The samples are independent.
Each sample is from a normally distributed population.
The population standard deviations of the groups are all equal. This property is known as homoscedasticity.
If these assumptions are not true for a given set of data, it may still be possible to use the Kruskal-Wallis H-test (scipy.stats.kruskal) although with some loss of power.
'''
effort = {'no effort': 1, 'low effort': 2.5, 'medium effort': 5, 'high effort': 7.5, 'maximum effort': 10}
rules = ['no effort', 'low effort', 'medium effort', 'high effort', 'maximum effort']
"""
Iyengar and Lepper, 1999
Task 1.1
Anglo American students:
PC: M = 7.39, SD = 1.88
EC: M = 3.06, SD = 1.89
MC: M = 2.94, SD = 1.84

Asian American students:
PC: M = 6.47, SD = 2.10
EC: M = 4.28, SD = 2.65
MC: M = 8.78, SD = 2.24
"""
print('Iyengar and Lepper, 1999')

statesIyengar = ['personal choice', 'out-group choice', 'in-group choice']

statesVectorIyengar = cu.encoder(statesIyengar)

personTypesIyengar = ['Anglo American', 'Asian American']

nParticipantsIyengar = 51#00

groupSizesIyengar = [int(nParticipantsIyengar/len(statesIyengar)), int(nParticipantsIyengar/len(statesIyengar))]

drivesIyengar = ['Achievement', 'Autonomy']#, 'Deference', 'Similance']

# Handpicked values. This part needs some evidence or something.
#                                       Ach  Aut  Def
stateAndDriveToSatisfactionIyengar = [[[.7,   .0,  .0],   # personal choice
                                       [.7, -0.7,  .2],   # out-group choice
                                       [.7, -0.7, 1.0]]]  # in-group choice

# 4.2.4.1 Initialization module. Handpicked values. This part needs some evidence or something.
#                                      Ach Aut Def
angloAmericanPersonToDeficitMapping = [.7, .8, .1]
angloAmericanPersonToDeficitMapping = cu.normalDist(angloAmericanPersonToDeficitMapping, nParticipantsIyengar, sigma=.15)

#                                      Ach Aut Def
asianAmericanPersonToDeficitMapping = [.6, .3, .5]
asianAmericanPersonToDeficitMapping = cu.normalDist(asianAmericanPersonToDeficitMapping, nParticipantsIyengar, sigma=.15)

# 4.2.4.2 Preprocessing module. Handpicked values. This part needs some evidence or something.
#                                 Ach  Aut Def
stateToStimulusMappingIyengar = [[.7,  .1, .1],   # personal choice
                                 [.5,  .7, .1],   # out-group choice
                                 [.7,  .6, .6]]   # in-group choice

# # 4.3.2.1 Goal Module. Handpicked values. This part needs some evidence or something.
# #                                        Ach Aut Def
# driveAndStateToRelevanceAnglIyengar = [[[.9, .7, .1],   # personal choice
#                                         [.9, .1, .1],   # out-group choice
#                                         [.9, .2, .5]]]  # in-group choice

'''
Step 5. Obtaining and Plotting the Results of the Rule Competition Through Utility

ANOVA in Iyengar and Lepper, 1999
Ethnicity: F(1,99) = 24.33, p < 0.0001
Condition: F(2,99) = 21.77, p < 0.0001
'''


IyengarAngloAmericanRES = cu.getResults(0, groupSizesIyengar,
                                      states=statesVectorIyengar,
                                      driveAndStateToRelevance=None,
                                      stateAndDriveToSatisfaction=stateAndDriveToSatisfactionIyengar,
                                      stateToStimulusMapping=stateToStimulusMappingIyengar,
                                      deficits=angloAmericanPersonToDeficitMapping, rules=rules, goals=['just do it'], effort=effort)
IyengarAsianAmericanRES = cu.getResults(1, groupSizesIyengar,
                                      states=statesVectorIyengar,
                                      driveAndStateToRelevance=None,
                                      stateAndDriveToSatisfaction=stateAndDriveToSatisfactionIyengar,
                                      stateToStimulusMapping=stateToStimulusMappingIyengar,
                                      deficits=asianAmericanPersonToDeficitMapping, rules=rules, goals=['just do it'], effort=effort)

score_angl_a, sd, se = cu.calcPerformance(IyengarAngloAmericanRES, 1.5)
print('Iyengar AngloAmerican M =', score_angl_a)
print('Iyengar AngloAmerican SD =', sd)
print('Iyengar AngloAmerican SE =', se)
score_asia_a, sd, se = cu.calcPerformance(IyengarAsianAmericanRES, 1.5)
print('Iyengar AsianAmerican SD =', sd)
print('Iyengar AsianAmerican SE =', se)
print('Iyengar AsianAmerican M =', score_asia_a)

IyengarConditionRES = np.array(IyengarAsianAmericanRES) + np.array(IyengarAngloAmericanRES)
IyengarEthnicityRES = np.array([np.sum(np.array(IyengarAngloAmericanRES), axis=0), np.sum(np.array(IyengarAsianAmericanRES), axis=0)])
print('!',np.sum(IyengarConditionRES), np.sum(IyengarEthnicityRES))

print('ANOVA Iyengar Condition', cu.anova(cu.resToFreq(IyengarConditionRES)))
print('ANOVA Iyengar Ethnicity', cu.anova(cu.resToFreq(IyengarEthnicityRES)))

print(cu.t_test(states=statesIyengar, res=IyengarConditionRES))
print(cu.t_test(states=personTypesIyengar, res=IyengarEthnicityRES))

cu.plot_results(personTypesIyengar, [score_angl_a, score_asia_a], statesIyengar, 'Iyengar')












'''
Deci, 1971
'''

print('\nDeci, 1971')

statesDeci = ['NO MR', 'MR INTRO', 'MR RESCIND']

statesVectorDeci = cu.encoder(statesDeci)
controlGroupStatesVecDeci = [statesVectorDeci[0], statesVectorDeci[0], statesVectorDeci[0]]

personTypesDeci = ['Control Group', 'Experiment Group']

nParticipantsDeci = 24

groupSizesDeci = [nParticipantsDeci, nParticipantsDeci]

drivesDeci = ['Achievement', 'Curiosity', 'Conservation']

#                                    Ach  Cur Con
stateAndDriveToSatisfactionDeci = [[[.7,  .7, .3],   # initial
                                    [.7,  .7, .7],   # monetary reward
                                    [.6,  .5, .2]]]  # m.r. is no longer offered

#                             Ach Cur Con
PersonToDeficitMappingDeci = [.7, .5, .7]
controlPersonToDeficitMappingDeci = cu.normalDist(PersonToDeficitMappingDeci, nParticipantsDeci, sigma=.01)
experimentPersonToDeficitMappingDeci = cu.normalDist(PersonToDeficitMappingDeci, nParticipantsDeci, sigma=.01)

#                              Ach Cur Con
stateToStimulusMappingDeci = [[.7, .7, .3],  # initial
                              [.5, .5, .8],  # monetary reward
                              [.5, .5, .2]]  # m.r. is no longer offered

# #                                 Ach Cur Con
# driveAndStateToRelevanceDeci = [[[.9, .9, .9],   # initial
#                                  [.9, .9, .9],   # monetary reward
#                                  [.9, .9, .9]]]  # m.r. is no longer offered



DeciControlGroupRES = cu.getResults(0, groupSizesDeci,
                                   states=statesVectorDeci,
                                   driveAndStateToRelevance= None,#[driveAndStateToRelevanceDeci[0],
                                                             #driveAndStateToRelevanceDeci[0],
                                                             #driveAndStateToRelevanceDeci[0]],
                                   stateAndDriveToSatisfaction=[[stateAndDriveToSatisfactionDeci[0][0],
                                                                stateAndDriveToSatisfactionDeci[0][0],
                                                                stateAndDriveToSatisfactionDeci[0][0]]],
                                   stateToStimulusMapping=[stateToStimulusMappingDeci[0],
                                                           stateToStimulusMappingDeci[0],
                                                           stateToStimulusMappingDeci[0],],
                                   deficits=controlPersonToDeficitMappingDeci, rules=rules, goals=['just do it'], effort=effort)


DeciExperimentalGroupRES = cu.getResults(1, groupSizesDeci,
                                        states=statesVectorDeci,
                                        driveAndStateToRelevance=None,#driveAndStateToRelevanceDeci,
                                        stateAndDriveToSatisfaction=stateAndDriveToSatisfactionDeci,
                                        stateToStimulusMapping=stateToStimulusMappingDeci,
                                        deficits=experimentPersonToDeficitMappingDeci, rules=rules, goals=['just do it'], effort=effort)


score_control_g, sd, _ = cu.calcPerformance(DeciControlGroupRES, 50)
score_experimental_g, sd2, _ = cu.calcPerformance(DeciExperimentalGroupRES, 50)

print('Deci ExperimentalGroup M =', score_experimental_g)
# print('Deci ExperimentalGroup SD =', sd2)#, se)
print('E(T3-T1)', score_experimental_g[2]-score_experimental_g[0])

print('Deci ControlGroup M =', score_control_g)
# print('Deci ControlGroup SD =', sd)#, se)
print('C(T3-T1)', score_control_g[2]-score_control_g[0])

print('E(T3-T1)-C(T3-T1)', score_experimental_g[2]-score_experimental_g[0]-(score_control_g[2]-score_control_g[0]))

DeciConditionRES = np.array(DeciControlGroupRES) + np.array(DeciExperimentalGroupRES)
print(DeciControlGroupRES)

DeciGroupRES = np.array([np.sum(np.array(DeciControlGroupRES), axis=0), np.sum(np.array(DeciExperimentalGroupRES), axis=0)])

print(cu.t_test(states=statesDeci, res=DeciConditionRES, df=48))
print(cu.t_test(states=personTypesDeci, res=DeciGroupRES, df=48))

cu.plot_results(personTypesDeci, [score_control_g, score_experimental_g], statesDeci, 'Deci')












'''
Stajkovic & Locke, 2006
'''

print('\nStajkovic & Locke, 2006')

statesStajkovic = ['P_E', 'P_DYB', 'P_D', 'NP_E', 'NP_DYB', 'NP_D']

statesVectorStajkovic = cu.encoder(statesStajkovic)

personTypesStajkovic = ['A Group']

nParticipantsStajkovic = int(78/6)

groupSizesStajkovic = [nParticipantsStajkovic]

drivesStajkovic = ['Achievement', 'Honor']

#                                     Ach  Hon
stateAndDriveToSatisfactionStajkovic = [[[.5,  .2],  # PE
                                         [.8,  .7],  # PDYB
                                         [.7,  .9],  # PD

                                         [.5,  .5],   # NPE
                                         [.7,  .6],   # NPDYB
                                         [.8,  .8]]]  # NPD

#                                            Ach Hon
StajkovicExperimentPersonToDeficitMapping = [.7, .6]
StajkovicExperimentPersonToDeficitMapping = cu.normalDist(StajkovicExperimentPersonToDeficitMapping, nParticipantsStajkovic, sigma=.01)


#                                   Ach Hon
stateToStimulusMappingStajkovic = [[.6, .4],  # PE
                                   [.6, .6],  # PDYB
                                   [.7, .7],  # PD

                                   [.5, .4],  # NPE
                                   [.4, .4],  # NPDYB
                                   [.6, .4]]  # NPD


#                                      Ach Hon
# driveAndStateToRelevanceStajkovic = [[[.9, .8],  # PE
#                                       [.9, .9],  # PDYB
#                                       [.9, .6],  # PD
#
#                                       [.9, .6],   # NPE
#                                       [.9, .7],   # NPDYB
#                                       [.9, .8]]]  # NPD

StajkovicRES = cu.getResults(id=0, groupSizes=groupSizesStajkovic,
                          states=statesVectorStajkovic,
                          driveAndStateToRelevance=None,#driveAndStateToRelevanceStajkovic,
                          stateAndDriveToSatisfaction=stateAndDriveToSatisfactionStajkovic,
                          stateToStimulusMapping=stateToStimulusMappingStajkovic,
                          deficits=StajkovicExperimentPersonToDeficitMapping, rules=rules, goals=['just do it'], effort=effort)

score_Stajkovic, sd_Stajkovic, se_Stajkovic = cu.calcPerformance(StajkovicRES, 1.1)
print('Stajkovic M = ', score_Stajkovic)
print('Stajkovic SD = ', sd_Stajkovic)

StajkovicPNPRES = np.array([np.sum(StajkovicRES[:3], axis=0), np.sum(StajkovicRES[3:], axis=0)])
StajkovicEDYBDRES = np.array(StajkovicRES[3:]) + np.array(StajkovicRES[:3])

# ANOVA in Stajkovic
# Prime, No Prime: F(1, 56) = 4.61, p < 0.05
# E, DYB, D: F(2, 55) = 6.46, p < 0.01

anova_StajkovicPNP = cu.anova(cu.resToFreq(StajkovicPNPRES))
anova_StajkovicEDYBD = cu.anova(cu.resToFreq(StajkovicEDYBDRES))

print(cu.t_test(states=statesStajkovic, res=StajkovicRES))

print('ANOVA Stajkovic Prime vs No Prime', anova_StajkovicPNP)
print('ANOVA Stajkovic Difficulty', anova_StajkovicEDYBD)


cu.plot_results(personTypesStajkovic, [score_Stajkovic], statesStajkovic, 'Stajkovic')






'''
Schmidt & DeShon, 2007
'''

print('\nSchmidt & DeShon, 2007')

effortSchmidt = {'referent': 2, 'nonreferent': 2, 'no effort': 0.01}
rulesSchmidt = ['referent', 'nonreferent', 'no effort']

statesSchmidt = ['R', 'NR']

statesVectorSchmidt = cu.encoder(statesSchmidt)

personTypesSchmidt = ['A Group']

nParticipantsSchmidt = int(253/6)

groupSizesSchmidt = [1]

drivesSchmidt = ['Achievement', 'Conservation']


#                                          Ach Con
SchmidtExperimentPersonToDeficitMapping = [.7, .8]
SchmidtExperimentPersonToDeficitMapping = cu.normalDist(SchmidtExperimentPersonToDeficitMapping, nParticipantsSchmidt, sigma=.01)


#                                 Ach Hon
stateToStimulusMappingSchmidt = [[.6, .4],  # R
                                 [.6, .4]]  # NR

SchmidtRESSummary = np.zeros(shape=[3])
time_spent_b = np.zeros(shape=nParticipantsSchmidt)
time_spent_w = np.zeros(shape=nParticipantsSchmidt)

for i in range(nParticipantsSchmidt):
    #                                      Ref
    #                                    Ach Hon
    driveAndStateToRelevanceSchmidt = [[[.9, .8],   # R
                                        [.0, .0]],  # NR
                                       # Non Ref
                                       [[.0, .0],   # R
                                        [.9, .8]],  # NR
                                       # Do Nothing
                                       [[.1, .2],   # R
                                        [.1, .2]]]  # NR

    #                                       Ach  Con
    stateAndDriveToSatisfactionSchmidt = [[[.7, .7],  # R
                                           [.7, .7]],  # NR

                                          [[.7, .7],  # R
                                           [.7, .7]],  # NR

                                          [[.5, .5],  # R
                                           [.5, .5]]]  # NR

    RorNR = randint(0, 1)
    worse_i = 0
    better_i = 0
    time_spent_b_counter = 0
    time_spent_w_counter = 0

    SchmidtRES_ = np.zeros(shape=[3])
    SchmidtRES_first10 = np.zeros(shape=[3])
    SchmidtRES_last10 = np.zeros(shape=[3])

    for min in range(38):


        if SchmidtRESSummary[0] >= 15:
            stateAndDriveToSatisfactionSchmidt[0][0][1] = 0
            stateAndDriveToSatisfactionSchmidt[0][1][1] = 0

        if SchmidtRESSummary[1] >= 15:
            stateAndDriveToSatisfactionSchmidt[1][0][1] = 0
            stateAndDriveToSatisfactionSchmidt[1][1][1] = 0

        SchmidtRES = cu.getResults(id=0, groupSizes=groupSizesSchmidt,
                                  states=[statesVectorSchmidt[RorNR]],
                                  driveAndStateToRelevance=driveAndStateToRelevanceSchmidt,
                                  stateAndDriveToSatisfaction=stateAndDriveToSatisfactionSchmidt,
                                  stateToStimulusMapping=stateToStimulusMappingSchmidt,
                                  deficits=SchmidtExperimentPersonToDeficitMapping, rules=rulesSchmidt,
                                  goals=['just do it'], effort=effortSchmidt)

        SchmidtRES_ += SchmidtRES[0]

        if min < 11:
            SchmidtRES_first10 += SchmidtRES[0]

        if min > 38 - 11:
            SchmidtRES_last10 += SchmidtRES[0]

        action = np.argmax(SchmidtRES)

        if action == 0:
            stateAndDriveToSatisfactionSchmidt[0][0][1] += -.025
            stateAndDriveToSatisfactionSchmidt[1][0][1] += -.025

        if action == 1:
            stateAndDriveToSatisfactionSchmidt[0][1][1] += -.025
            stateAndDriveToSatisfactionSchmidt[1][1][1] += -.025

        if action < 2:
            RorNR = action


        if SchmidtRESSummary[0] > SchmidtRESSummary[1]:
            better_i = 0
            worse_i = 1
        elif SchmidtRESSummary[1] > SchmidtRESSummary[0]:
            better_i = 1
            worse_i = 0

        if better_i != worse_i:
            if RorNR == better_i:
                time_spent_b_counter += 1
            else:
                time_spent_w_counter += 1

    if SchmidtRES_[0] > 14:
        SchmidtRESSummary[0] += 1  # R

    if SchmidtRES_[1] > 14:
        SchmidtRESSummary[1] += 1  # NR

    if SchmidtRES_[0] > 14 and SchmidtRES_[1] > 14:
        SchmidtRESSummary[2] += 1  # Both

    time_spent_b[i] = time_spent_b_counter
    time_spent_w[i] = time_spent_w_counter

# print(time_spent_b, time_spent_w)
f10_total = np.sum(time_spent_b[:10] + time_spent_w[:10])
f10_b = np.sum(time_spent_b[:10])
f10_w = np.sum(time_spent_w[:10])

l10_total = np.sum(time_spent_b[-10:] + time_spent_w[-10:])
l10_b = np.sum(time_spent_b[-10:])
l10_w = np.sum(time_spent_w[-10:])

print('Figure 3', f10_w/f10_total, f10_b/f10_total, l10_w/l10_total, l10_b/l10_total)

print('Schmidt REF, NONREF, BOTH (in %)', SchmidtRESSummary/nParticipantsSchmidt*100)







'''
Seijts & Latham, 2001
'''

print('\nSeijts & Latham, 2001')


statesSeijts = ['Outcome', 'Learning', 'DYB']

statesVectorSeijts = cu.encoder(statesSeijts)

personTypesSeijts = ['A Group']

nParticipantsSeijts = int((62+32)/3)

groupSizesSeijts = [nParticipantsSeijts]


drivesSeijts = ['Achievement', 'Honor', 'Curiosity']

#                                      Ach  Hon Cur
stateAndDriveToSatisfactionSeijts = [[[.7, -.3, .1],   # O
                                      [.7,  .2, .6],   # L
                                      [.7,  .0, .6]]]  # DYB

#                                            Ach Hon
SeijtsExperimentPersonToDeficitMapping = [.7, .7, .7]
SeijtsExperimentPersonToDeficitMapping = cu.normalDist(SeijtsExperimentPersonToDeficitMapping, nParticipantsSeijts, sigma=0.1)


#                                Ach Hon Cur
stateToStimulusMappingSeijts = [[.8, .6, .1],  # O
                                [.8, .1, .5],  # L
                                [.5, .0, .5]]  # DYB


# #                                   Ach Hon Cur
# driveAndStateToRelevanceSeijts = [[[.9, .7, .8],   # O
#                                    [.9, .7, .8],   # L
#                                    [.9, .7, .8]]]  # DYB

SeijtsRES = cu.getResults(id=0, groupSizes=groupSizesSeijts, states=statesVectorSeijts,
                        driveAndStateToRelevance=None,#driveAndStateToRelevanceSeijts,
                        stateAndDriveToSatisfaction=stateAndDriveToSatisfactionSeijts,
                        stateToStimulusMapping=stateToStimulusMappingSeijts,
                        deficits=SeijtsExperimentPersonToDeficitMapping, rules=rules, goals=['just do it'],
                        effort=effort)

score_Seijts, sd_Seijts, se_Seijts = cu.calcPerformance(SeijtsRES, 1.825)
print('Seijts M =', score_Seijts)
print('Seijts SD =', sd_Seijts)

anova_SeijtsOLDYB = cu.anova(cu.resToFreq(SeijtsRES))
print('ANOVA Seijts Outcome vs Learning vs DYB:', anova_SeijtsOLDYB)
print(cu.t_test(states=statesSeijts, res=SeijtsRES))

cu.plot_results(personTypesSeijts, [score_Seijts], statesSeijts, 'Seijts')








'''
Chen & Latham, 2014
M = 9.28, 7.09, 7.71, 6.69
'''

print('\nChen & Latham, 2014')

statesChen = ['Learning', 'Performance', 'Learning & Performance', 'Control']

statesVectorChen = cu.encoder(statesChen)

personTypesChen = ['A Group']

nParticipantsChen = 22

groupSizesChen = [nParticipantsChen]


drivesChen = ['Achievement', 'Honor']

#                                    Ach  Hon
stateAndDriveToSatisfactionChen = [[[1.,  .0],   # L
                                    [1., -.55],  # P
                                    [1., -.3],   # PL
                                    [1.,  .0]]]  # C

#                                       Ach Hon
ChenExperimentPersonToDeficitMapping = [.7, .7]
ChenExperimentPersonToDeficitMapping = cu.normalDist(ChenExperimentPersonToDeficitMapping, nParticipantsChen)


#                              Ach  Hon
stateToStimulusMappingChen = [[1.,  .1],  # L
                              [.8,  .7],  # P
                              [.62, .4],  # PL
                              [.4,  .1]]  # C


#                                 Ach Hon
# driveAndStateToRelevanceChen = [[[.9, .7],   # L
#                                  [.9, .7],   # P
#                                  [.9, .7],   # PL
#                                  [.9, .7]]]  # C

ChenRES = cu.getResults(id=0, groupSizes=groupSizesChen, states=statesVectorChen,
                        driveAndStateToRelevance=None,#driveAndStateToRelevanceChen,
                        stateAndDriveToSatisfaction=stateAndDriveToSatisfactionChen,
                        stateToStimulusMapping=stateToStimulusMappingChen,
                        deficits=ChenExperimentPersonToDeficitMapping, rules=rules, goals=['just do it'],
                        effort=effort)

score_Chen, sd_Chen, se_Chen = cu.calcPerformance(ChenRES, 1.93)
print('Chen M =', score_Chen)
print('Chen SD =', sd_Chen)

anova_Chen = cu.anova(cu.resToFreq(ChenRES))
print('ANOVA Chen Condition:', anova_Chen)
print(cu.t_test(states=statesChen, res=ChenRES))

cu.plot_results(personTypesChen, [score_Chen], statesChen, 'Chen')






'''
Cianci, Klein & Seijts (2010)
M = 9.28, 7.09, 7.71, 6.69
'''

print('\nCianci, Klein & Seijts (2010)')


statesCianci = ['Learning', 'Performance']

statesVectorCianci = cu.encoder(statesCianci)

personTypesCianci = ['Low Conscientiousness Group', 'High Conscientiousness Group']

nParticipantsCianci = 37

groupSizesCianci = [nParticipantsCianci, nParticipantsCianci]


drivesCianci = ['Achievement', 'Honor']

#                                     Ach   Hon
stateAndDriveToSatisfactionCianci = [[[1.,  .0],   # L
                                      [1., -.8]]]  # P

#                                     Ach Hon
CianciLowConPersonToDeficitMapping = [.7, .6]
CianciLowConPersonToDeficitMapping = cu.normalDist(CianciLowConPersonToDeficitMapping, nParticipantsCianci)

#                                      Ach Hon
CianciHighConPersonToDeficitMapping = [.9, .9]
CianciHighConPersonToDeficitMapping = cu.normalDist(CianciHighConPersonToDeficitMapping, nParticipantsCianci)


#                                Ach Hon
stateToStimulusMappingCianci = [[1., .4],  # L
                                [.7, .9]]  # P


# #                                   Ach Hon
# driveAndStateToRelevanceCianci = [[[.9, .7],   # L
#                                    [.9, .7]]]  # P

CianciLowC_RES = cu.getResults(id=0, groupSizes=groupSizesCianci, states=statesVectorCianci,
                        driveAndStateToRelevance=None,#driveAndStateToRelevanceCianci,
                        stateAndDriveToSatisfaction=stateAndDriveToSatisfactionCianci,
                        stateToStimulusMapping=stateToStimulusMappingCianci,
                        deficits=CianciLowConPersonToDeficitMapping, rules=rules, goals=['just do it'],
                        effort=effort)

CianciHighC_RES = cu.getResults(id=0, groupSizes=groupSizesCianci, states=statesVectorCianci,
                        driveAndStateToRelevance=None,#driveAndStateToRelevanceCianci,
                        stateAndDriveToSatisfaction=stateAndDriveToSatisfactionCianci,
                        stateToStimulusMapping=stateToStimulusMappingCianci,
                        deficits=CianciHighConPersonToDeficitMapping, rules=rules, goals=['just do it'],
                        effort=effort)

score_CianciL, sd_Cianci, _ = cu.calcPerformance(CianciLowC_RES, 2.3)
print('Cianci Low C. M =', score_CianciL)
print('Cianci Low C. SD =', sd_Cianci)

score_CianciH, sd_Cianci, _ = cu.calcPerformance(CianciHighC_RES, 2.3)
print('Cianci High C. M =', score_CianciH)
print('Cianci High C. SD =', sd_Cianci)

anova_Cianci1 = cu.anova(cu.resToFreq(CianciLowC_RES+CianciHighC_RES))
print('ANOVA Cianci Outcome vs Learning:', anova_Cianci1)

anova_Cianci2 = cu.anova(cu.resToFreq([CianciLowC_RES[0]+CianciLowC_RES[1], CianciHighC_RES[0]+CianciHighC_RES[1]]))
print('ANOVA Cianci Low Conscientiousness vs High Conscientiousness:', anova_Cianci2)

cu.plot_results(personTypesCianci, [score_CianciL, score_CianciH], statesCianci, 'Cianci')






'''
Shih & Alexander (2000)
M = 9.28, 7.09, 7.71, 6.69
'''

print('\nShih & Alexander (2000)')

statesShih = ['Goal_Self', 'Goal_Social', 'No_Goal_Self', 'No_Goal_Social']

statesVectorShih = cu.encoder(statesShih)

personTypesShih = ['A Group']

nParticipantsShih = 21

groupSizesShih = [nParticipantsShih]

drivesShih = ['Achievement', 'Honor', 'Avoid Unpleasantness']

#                                    Ach  Hon  AUn
stateAndDriveToSatisfactionShih = [[[.7,  .0,  .0],   # GSe
                                    [.7, -.2, -.1],   # GSo
                                    [.7,  .0,  .0],   # NGSe
                                    [.7, -.3, -.3]]]  # NGSo

#                                       Ach Hon AUn
ShihExperimentPersonToDeficitMapping = [.7, .7, .7]
ShihExperimentPersonToDeficitMapping = cu.normalDist(ShihExperimentPersonToDeficitMapping, nParticipantsShih)


#                              Ach Hon  AUn
stateToStimulusMappingShih = [[.7, .3,  .1],   # GSe
                              [.7, .2,  .3],   # GSo
                              [.7, .3,  .05],  # NGSe
                              [.7, .3,  .2]]   # NGSo


# #                                 Ach Hon  AUn
# driveAndStateToRelevanceShih = [[[.7,  .3, .1],   # GSe
#                                  [.7,  .2, .6],   # GSo
#                                  [.7,  .2, .6],   # NGSe
#                                  [.7,  .0, .6]]]  # NGSo

ShihRES = cu.getResults(id=0, groupSizes=groupSizesShih, states=statesVectorShih,
                        driveAndStateToRelevance=None,#driveAndStateToRelevanceShih,
                        stateAndDriveToSatisfaction=stateAndDriveToSatisfactionShih,
                        stateToStimulusMapping=stateToStimulusMappingShih,
                        deficits=ShihExperimentPersonToDeficitMapping, rules=rules, goals=['just do it'],
                        effort=effort)


score_Shih, sd_Shih, se_Shih = cu.calcPerformance(ShihRES, 14)
print('Shih M =', score_Shih)
print('Shih SD =', sd_Shih)

anova_Shih = cu.anova(cu.resToFreq(ShihRES))
print('ANOVA Shih Goal Self vs Goal Social vs No Goal Self vs No Goal Social:', anova_Shih)
print(cu.t_test(states=statesShih, res=ShihRES))

ShihRES_GvsNG = np.array([np.sum(ShihRES[:2], axis=0), np.sum(ShihRES[2:], axis=0)])

anova_Shih_GvsNG = cu.anova(cu.resToFreq(ShihRES_GvsNG))
print('ANOVA Shih Goal vs No Goal:', anova_Shih_GvsNG)

cu.plot_results(personTypesChen, [score_Chen], statesChen, 'Shih')

