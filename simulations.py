import utils as u
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
# """
# Iyengar and Lepper, 1999
# Task 1.1
# Anglo American students:
# PC: M = 7.39, SD = 1.88
# EC: M = 3.06, SD = 1.89
# MC: M = 2.94, SD = 1.84
#
# Asian American students:
# PC: M = 6.47, SD = 2.10
# EC: M = 4.28, SD = 2.65
# MC: M = 8.78, SD = 2.24
# """
# print('Iyengar and Lepper, 1999')
#
# statesIyengar = ['personal choice', 'out-group choice', 'in-group choice']
#
# statesVectorIyengar = u.encoder(statesIyengar)
#
# personTypesIyengar = ['Anglo American', 'Asian American']
#
# nParticipantsIyengar = 51#00
#
# groupSizesIyengar = [int(nParticipantsIyengar/len(statesIyengar)), int(nParticipantsIyengar/len(statesIyengar))]
#
# drivesIyengar = ['Achievement', 'Autonomy']#, 'Deference', 'Similance']
#
# # Handpicked values. This part needs some evidence or something.
# #                                       Ach  Aut  Def
# stateAndDriveToSatisfactionIyengar = [[[.7,   .0,  .0],   # personal choice
#                                        [.7, -0.7,  .2],   # out-group choice
#                                        [.7, -0.7, 1.0]]]  # in-group choice
#
# # 4.2.4.1 Initialization module. Handpicked values. This part needs some evidence or something.
# #                                      Ach Aut Def
# # angloAmericanPersonToDeficitMapping = [.7, .8, .1]
# angloAmericanPersonToDeficitMapping = [.6, .8, .1]
# angloAmericanPersonToDeficitMapping = u.normalDist(angloAmericanPersonToDeficitMapping, nParticipantsIyengar, sigma=.1)
#
# #                                      Ach Aut Def
# asianAmericanPersonToDeficitMapping = [.6, .3, .5]
# asianAmericanPersonToDeficitMapping = u.normalDist(asianAmericanPersonToDeficitMapping, nParticipantsIyengar, sigma=.1)
#
# # 4.2.4.2 Preprocessing module. Handpicked values. This part needs some evidence or something.
# #                                 Ach  Aut Def
# # stateToStimulusMappingIyengar = [[.7,  .1, .1],   # personal choice
# #                                  [.5,  .7, .1],   # out-group choice
# #                                  [.7,  .6, .6]]   # in-group choice
# stateToStimulusMappingIyengar = [[.7,  .1, .1],   # personal choice
#                                  [.7,  .6, .1],   # out-group choice
#                                  [.7,  .6, .6]]   # in-group choice
#
# # # 4.3.2.1 Goal Module. Handpicked values. This part needs some evidence or something.
# # #                                        Ach Aut Def
# # driveAndStateToRelevanceAnglIyengar = [[[.9, .7, .1],   # personal choice
# #                                         [.9, .1, .1],   # out-group choice
# #                                         [.9, .2, .5]]]  # in-group choice
#
# '''
# Step 5. Obtaining and Plotting the Results of the Rule Competition Through Utility
#
# ANOVA in Iyengar and Lepper, 1999
# Ethnicity: F(1,99) = 24.33, p < 0.0001
# Condition: F(2,99) = 21.77, p < 0.0001
# '''
#
#
# IyengarAngloAmericanRES = u.getResults(0, groupSizesIyengar,
#                                       states=statesVectorIyengar,
#                                       driveAndStateToRelevance=None,
#                                       stateAndDriveToSatisfaction=stateAndDriveToSatisfactionIyengar,
#                                       stateToStimulusMapping=stateToStimulusMappingIyengar,
#                                       deficits=angloAmericanPersonToDeficitMapping, rules=rules, goals=['just do it'], effort=effort)
# IyengarAsianAmericanRES = u.getResults(1, groupSizesIyengar,
#                                       states=statesVectorIyengar,
#                                       driveAndStateToRelevance=None,
#                                       stateAndDriveToSatisfaction=stateAndDriveToSatisfactionIyengar,
#                                       stateToStimulusMapping=stateToStimulusMappingIyengar,
#                                       deficits=asianAmericanPersonToDeficitMapping, rules=rules, goals=['just do it'], effort=effort)
#
# score_angl_a, sd, se = u.calcPerformance(IyengarAngloAmericanRES, 1.5)
# print('Iyengar AngloAmerican M =', score_angl_a)
# print('Iyengar AngloAmerican SD =', sd)
# # print('Iyengar AngloAmerican SE =', se)
# score_asia_a, sd, se = u.calcPerformance(IyengarAsianAmericanRES, 1.5)
# print('Iyengar AsianAmerican SD =', sd)
# # print('Iyengar AsianAmerican SE =', se)
# print('Iyengar AsianAmerican M =', score_asia_a)
#
# # print('!', IyengarAsianAmericanRES)
# IyengarInteractionRES = np.concatenate((np.array(IyengarAsianAmericanRES), np.array(IyengarAngloAmericanRES)), axis=0)
#
# IyengarConditionRES = np.array(IyengarAsianAmericanRES) + np.array(IyengarAngloAmericanRES)
# IyengarEthnicityRES = np.array([np.sum(np.array(IyengarAngloAmericanRES), axis=0), np.sum(np.array(IyengarAsianAmericanRES), axis=0)])
# # print('!',np.sum(IyengarConditionRES), np.sum(IyengarEthnicityRES))
#
# print('ANOVA Iyengar Condition', u.anova(u.resToFreq(IyengarConditionRES)))
# print('ANOVA Iyengar Ethnicity', u.anova(u.resToFreq(IyengarEthnicityRES)))
# print('ANOVA Iyengar Interaction', u.anova(u.resToFreq(IyengarInteractionRES)))
#
# print('t-test for Asian American')
# print(u.t_test(states=statesIyengar, res=IyengarAsianAmericanRES))
# print('t-test for Anglo American')
# print(u.t_test(states=statesIyengar, res=IyengarAngloAmericanRES))
# print('t-test for Ethnicity')
# print(u.t_test(states=personTypesIyengar, res=IyengarEthnicityRES))
#
# u.plot_results(personTypesIyengar, [score_angl_a, score_asia_a], statesIyengar, 'Iyengar')
#
#
#
#
#
#
#
#
#
#
#
#
# '''
# Deci, 1971
# '''
#
# print('\nDeci, 1971')
#
# statesDeci = ['NO MR', 'MR INTRO', 'MR RESCIND']
#
# statesVectorDeci = u.encoder(statesDeci)
# controlGroupStatesVecDeci = [statesVectorDeci[0], statesVectorDeci[0], statesVectorDeci[0]]
#
# personTypesDeci = ['Control Group', 'Experiment Group']
#
# nParticipantsDeci = 12
#
# groupSizesDeci = [nParticipantsDeci, nParticipantsDeci]
#
# drivesDeci = ['Achievement', 'Curiosity', 'Conservation']
#
# #                                    Ach  Cur Con
# stateAndDriveToSatisfactionDeci = [[[.7,  .7, -.7],   # initial
#                                     [.7,  .7, .0],   # monetary reward
#                                     [.6,  .5, -.1]]]  # m.r. is no longer offered
#
# #                             Ach Cur Con
# PersonToDeficitMappingDeci = [.7, .7, .6]
# controlPersonToDeficitMappingDeci = u.normalDist(PersonToDeficitMappingDeci, nParticipantsDeci, sigma=.01)
# experimentPersonToDeficitMappingDeci = u.normalDist(PersonToDeficitMappingDeci, nParticipantsDeci, sigma=.01)
#
# #                              Ach Cur Con
# stateToStimulusMappingDeci = [[.7, .7, .3],  # initial
#                               [.5, .5, .2],  # monetary reward
#                               [.5, .5, .8]]  # m.r. is no longer offered
#
# # #                                 Ach Cur Con
# # driveAndStateToRelevanceDeci = [[[.9, .9, .9],   # initial
# #                                  [.9, .9, .9],   # monetary reward
# #                                  [.9, .9, .9]]]  # m.r. is no longer offered
#
#
#
# DeciControlGroupRES = u.getResults(0, groupSizesDeci,
#                                    states=statesVectorDeci,
#                                    driveAndStateToRelevance= None,#[driveAndStateToRelevanceDeci[0],
#                                                              #driveAndStateToRelevanceDeci[0],
#                                                              #driveAndStateToRelevanceDeci[0]],
#                                    stateAndDriveToSatisfaction=[[stateAndDriveToSatisfactionDeci[0][0],
#                                                                 stateAndDriveToSatisfactionDeci[0][0],
#                                                                 stateAndDriveToSatisfactionDeci[0][0]]],
#                                    stateToStimulusMapping=[stateToStimulusMappingDeci[0],
#                                                            stateToStimulusMappingDeci[0],
#                                                            stateToStimulusMappingDeci[0],],
#                                    deficits=controlPersonToDeficitMappingDeci, rules=rules, goals=['just do it'], effort=effort)
#
#
# DeciExperimentalGroupRES = u.getResults(1, groupSizesDeci,
#                                         states=statesVectorDeci,
#                                         driveAndStateToRelevance=None,#driveAndStateToRelevanceDeci,
#                                         stateAndDriveToSatisfaction=stateAndDriveToSatisfactionDeci,
#                                         stateToStimulusMapping=stateToStimulusMappingDeci,
#                                         deficits=experimentPersonToDeficitMappingDeci, rules=rules, goals=['just do it'], effort=effort)
#
#
# score_control_g, sd, _ = u.calcPerformance(DeciControlGroupRES, 50)
# ec_freq_diference = [u.resToFreq(DeciExperimentalGroupRES)[2] - u.resToFreq(DeciExperimentalGroupRES)[0], u.resToFreq(DeciControlGroupRES)[2] - u.resToFreq(DeciControlGroupRES)[0]]
# # print([u.resToFreq(DeciExperimentalGroupRES)[2] - u.resToFreq(DeciExperimentalGroupRES)[0], u.resToFreq(DeciControlGroupRES)[2] - u.resToFreq(DeciControlGroupRES)[0]])
# print('E(T3-T1)-C(T3-T1) ANOVA:', u.anova(np.array(ec_freq_diference)))
# score_experimental_g, sd2, _ = u.calcPerformance(DeciExperimentalGroupRES, 50)
#
# print('Deci ExperimentalGroup M =', score_experimental_g)
# # print('Deci ExperimentalGroup SD =', sd2)#, se)
# print('E(T3-T1)', score_experimental_g[2]-score_experimental_g[0])
#
# print('Deci ControlGroup M =', score_control_g)
# # print('Deci ControlGroup SD =', sd)#, se)
# print('C(T3-T1)', score_control_g[2]-score_control_g[0])
#
# print('E(T3-T1)-C(T3-T1)', score_experimental_g[2]-score_experimental_g[0]-(score_control_g[2]-score_control_g[0]))
#
# DeciConditionRES = np.array(DeciControlGroupRES) + np.array(DeciExperimentalGroupRES)
#
# DeciGroupRES = np.array([DeciExperimentalGroupRES[0], DeciExperimentalGroupRES[2]])
#
# print(u.t_test(states=statesDeci, res=DeciConditionRES, df=47))
# print(u.t_test(states=personTypesDeci, res=DeciGroupRES, df=47))
#
# u.plot_results(personTypesDeci, [score_control_g, score_experimental_g], statesDeci, 'Deci')
#
#
#
#
#
#
#
#
#
#
#
#
# '''
# Stajkovic & Locke, 2006
# '''
#
# print('\nStajkovic & Locke, 2006')
#
# statesStajkovic = ['P_E', 'P_DYB', 'P_D', 'NP_E', 'NP_DYB', 'NP_D']
#
# statesVectorStajkovic = u.encoder(statesStajkovic)
#
# personTypesStajkovic = ['A Group']
#
# nParticipantsStajkovic = int(78/6)
#
# groupSizesStajkovic = [nParticipantsStajkovic]
#
# drivesStajkovic = ['Achievement', 'Honor']
#
# #                                         Ach   Hon
# stateAndDriveToSatisfactionStajkovic = [[[.5,  -.1],  # PE
#                                          [.8,  -.1],  # PDYB
#                                          [.9,  -.1],  # PD
#
#                                          [.5,  -.1],   # NPE
#                                          [.5,  -.1],   # NPDYB
#                                          [.7,  -.1]]]  # NPD
#
# #                                            Ach Hon
# StajkovicExperimentPersonToDeficitMapping = [.8, .5]
# StajkovicExperimentPersonToDeficitMapping = u.normalDist(StajkovicExperimentPersonToDeficitMapping, nParticipantsStajkovic, sigma=.01)
#
#
# #                                   Ach Hon
# stateToStimulusMappingStajkovic = [[.4, .4],  # PE
#                                    [.6, .6],  # PDYB
#                                    [.9, .5],  # PD
#
#                                    [.4, .4],  # NPE
#                                    [.5, .4],  # NPDYB
#                                    [.6, .4]]  # NPD
#
#
# #                                      Ach Hon
# # driveAndStateToRelevanceStajkovic = [[[.9, .8],  # PE
# #                                       [.9, .9],  # PDYB
# #                                       [.9, .6],  # PD
# #
# #                                       [.9, .6],   # NPE
# #                                       [.9, .7],   # NPDYB
# #                                       [.9, .8]]]  # NPD
#
# StajkovicRES = u.getResults(id=0, groupSizes=groupSizesStajkovic,
#                           states=statesVectorStajkovic,
#                           driveAndStateToRelevance=None,#driveAndStateToRelevanceStajkovic,
#                           stateAndDriveToSatisfaction=stateAndDriveToSatisfactionStajkovic,
#                           stateToStimulusMapping=stateToStimulusMappingStajkovic,
#                           deficits=StajkovicExperimentPersonToDeficitMapping, rules=rules, goals=['just do it'], effort=effort)
#
# score_Stajkovic, sd_Stajkovic, se_Stajkovic = u.calcPerformance(StajkovicRES, 1.3)
# print('Stajkovic M = ', score_Stajkovic)
# print('Stajkovic SD = ', sd_Stajkovic)
#
# StajkovicPNPRES = np.array([np.sum(StajkovicRES[:3], axis=0), np.sum(StajkovicRES[3:], axis=0)])
# StajkovicEDYBDRES = np.array(StajkovicRES[3:]) + np.array(StajkovicRES[:3])
#
# # ANOVA in Stajkovic
# # Prime, No Prime: F(1, 56) = 4.61, p < 0.05
# # E, DYB, D: F(2, 55) = 6.46, p < 0.01
#
# anova_StajkovicPNP = u.anova(u.resToFreq(StajkovicPNPRES))
# anova_StajkovicEDYBD = u.anova(u.resToFreq(StajkovicEDYBDRES))
#
# print(u.t_test(states=statesStajkovic, res=StajkovicRES))
#
# print('ANOVA Stajkovic Prime vs No Prime', anova_StajkovicPNP)
# print('ANOVA Stajkovic Difficulty', anova_StajkovicEDYBD)
#
#
# u.plot_results(personTypesStajkovic, [score_Stajkovic], statesStajkovic, 'Stajkovic')
#
#
#



'''
Schmidt & DeShon, 2007
'''

print('\nSchmidt & DeShon, 2007')

# effortSchmidt = {'referent': 2, 'nonreferent': 2, 'no effort': 0.01}
# rulesSchmidt = ['referent', 'nonreferent', 'no effort']

statesSchmidt = [['T']]

statesVectorSchmidt = u.encoder(statesSchmidt)
# print(statesVectorSchmidt)

personTypesSchmidt = ['A Group']
nParticipantsSchmidt = int(253/6)
print(nParticipantsSchmidt)

groupSizesSchmidt = [1]

drivesSchmidt = ['Achievement']


#                                          Ach
SchmidtExperimentPersonToDeficitMapping = [.8]



#                                 Ach
stateToStimulusMappingSchmidt = [[.6]]  # NR

SchmidtRESSummary = np.zeros(shape=[3])
time_spent_b = np.zeros(shape=nParticipantsSchmidt)
time_spent_w = np.zeros(shape=nParticipantsSchmidt)
f10_w_ = 0
f10_total_ = 0
f10_b_ = 0
l10_w_ = 0
l10_total_ = 0
l10_b_ = 0
f10_b_list = []
f10_w_list = []
l10_b_list = []
l10_w_list = []

for i in range(nParticipantsSchmidt):

    SchmidtExperimentPersonToDeficitMapping = u.normalDist(SchmidtExperimentPersonToDeficitMapping, 1, sigma=.1)
    # driveAndStateToRelevanceSchmidt = np.array([[[1]],  # R
    #                                             [[2]]])  # NR
    #
    # stateAndDriveToSatisfactionSchmidt = np.array([[[3]],  # R
    #                                                [[4]]])  # NR
    # print('!!', stateAndDriveToSatisfactionSchmidt[1][0][0],
    # driveAndStateToRelevanceSchmidt[1][0][0],
    # driveAndStateToRelevanceSchmidt[0][0][0],
    # stateAndDriveToSatisfactionSchmidt[0][0][0])
    driveAndStateToRelevanceSchmidt = np.array([[[.9]],  # R
                                                [[.9]]])  # NR

    stateAndDriveToSatisfactionSchmidt = np.array([[[.7]],  # R
                                                   [[.7]]])  # NR

    # RorNR = randint(0, 1)
    worse_i = 0
    better_i = 0
    time_spent_b_counter = []
    time_spent_w_counter = []

    SchmidtRES_ = np.zeros(shape=[2])
    SchmidtRES_first10 = np.zeros(shape=[3])
    SchmidtRES_last10 = np.zeros(shape=[3])

    timeAvaliable = 30
    j=0
    while timeAvaliable > 0:

        if SchmidtRES_[0] > SchmidtRES_[1]:
            better_i = 0
            worse_i = 1
        elif SchmidtRES_[1] > SchmidtRES_[0]:
            better_i = 1
            worse_i = 0
        elif SchmidtRES_[1] == SchmidtRES_[0]:
            better_i = 1
            worse_i = 1


        if SchmidtRES_[0] > SchmidtRES_[1]:  # R > NR
            driveAndStateToRelevanceSchmidt = [[[.1]],  # R
                                               [[.9]]]  # NR

            # stateAndDriveToSatisfactionSchmidt = np.array([[[.4]],  # R
            #                                                [[.7]]])  # NR

        if SchmidtRES_[0] < SchmidtRES_[1]:  # R < NR
            driveAndStateToRelevanceSchmidt = [[[.9]],  # R
                                               [[.1]]]  # NR

            # stateAndDriveToSatisfactionSchmidt = np.array([[[.7]],  # R
            #                                                [[.4]]])  # NR

        if SchmidtRES_[0] == SchmidtRES_[1]:  # R = NR
            driveAndStateToRelevanceSchmidt = [[[.9]],  # R
                                               [[.9]]]  # NR

            # stateAndDriveToSatisfactionSchmidt = np.array([[[.7]],  # R
            #                                                [[.7]]])  # NR


        # if SchmidtRES_[0] > 14 or ((15-SchmidtRES_[0]) > (timeAvaliable)):  # R > 15 or not enough time for R
        if  ((15-SchmidtRES_[0]) > (timeAvaliable)):  # R > 15 or not enough time for R
            # print('0 is pointless')
            driveAndStateToRelevanceSchmidt[0][0][0] = 0.0
            stateAndDriveToSatisfactionSchmidt[0][0][0] = 0

        # if SchmidtRES_[1] > 14 or ((15-SchmidtRES_[1]) > (timeAvaliable)):  # NR > 15 or not enough time for NR
        if ((15-SchmidtRES_[1]) > (timeAvaliable)):  # NR > 15 or not enough time for NR
            # print('1 is pointless', 15-SchmidtRES_[1], timeAvaliable, (15-SchmidtRES_[1]) > (timeAvaliable))
            stateAndDriveToSatisfactionSchmidt[1][0][0] = 0
            driveAndStateToRelevanceSchmidt[1][0][0] = 0.0

        SchmidtRES, SchmidtGoal = u.getResults(id=0, groupSizes=groupSizesSchmidt,
                                  states=statesVectorSchmidt,
                                  driveAndStateToRelevance=driveAndStateToRelevanceSchmidt,
                                  stateAndDriveToSatisfaction=stateAndDriveToSatisfactionSchmidt,
                                  stateToStimulusMapping=stateToStimulusMappingSchmidt,
                                  deficits=SchmidtExperimentPersonToDeficitMapping, rules=rules,
                                  goals=['r', 'nr'], effort=effort)


        temp = np.zeros(shape=[2])
        temp[SchmidtGoal] = 1.0
        SchmidtRES_ += temp

        action = SchmidtGoal
        aaa = [1, 2.5, 5, 7.5, 10]  # {'no e': 1, 'low e': 2.5, 'medium e': 5, 'high e': 7.5, 'maximum e': 10}
        timeSpent = .4*(aaa[list(SchmidtRES[0]).index(1)])  # "2.5", SchmidtRES[0].index(1) need to be effort levels
        timeAvaliable -= timeSpent
        if SchmidtRES_[0] > 15 and action == 0:
            timeSpent = 0
        if SchmidtRES_[1] > 15 and action == 1:
            timeSpent = 0


        if better_i != worse_i:
            if action == better_i:
                time_spent_b_counter.append(timeSpent)
                time_spent_w_counter.append(0)
            else:
                time_spent_b_counter.append(0)
                time_spent_w_counter.append(timeSpent)
        else:
            time_spent_b_counter.append(timeSpent/2)
            time_spent_w_counter.append(timeSpent/2)

        # print('Goal:', action,
        #       'Effort Level:', np.argmax(SchmidtRES),
        #       'Time Spent W/B:', [time_spent_w_counter[j], time_spent_b_counter[j]],
        #       'Schedules Soleved:', SchmidtRES_,
        #       'Time Spent Counter W/B:', [np.sum(time_spent_w_counter), np.sum(time_spent_b_counter)],
        #       # '', SchmidtGoal,
        #       'Better', better_i,
        #       'Worse', worse_i,
        #       '\nEnd of Turn\n')
        j += 1

    if SchmidtRES_[0] > 14:
        SchmidtRESSummary[0] += 1  # R

    if SchmidtRES_[1] > 14:
        SchmidtRESSummary[1] += 1  # NR

    if SchmidtRES_[0] > 14 and SchmidtRES_[1] > 14:
        SchmidtRESSummary[2] += 1  # Both

    # time_spent_b[i] = time_spent_b_counter # :10 HERE NOT ALTER!!!
    # time_spent_w[i] = time_spent_w_counter

    # print(time_spent_b, time_spent_w)
    f10_total = 0
    f10_b = 0
    f10_w = 0
    for step in range(len(time_spent_b_counter)):
        f10_total += time_spent_b_counter[step] + time_spent_w_counter[step]
        if f10_total <= 10:
            f10_b += time_spent_b_counter[step]
            f10_w += time_spent_w_counter[step]
        else:
            f10_total -= time_spent_b_counter[step] + time_spent_w_counter[step]
            break

    l10_total = 0
    l10_b = 0
    l10_w = 0
    for step in reversed(range(len(time_spent_b_counter))):
        l10_total += time_spent_b_counter[step] + time_spent_w_counter[step]
        if l10_total <= 10:
            l10_b += time_spent_b_counter[step]
            l10_w += time_spent_w_counter[step]
        else:
            l10_total -= time_spent_b_counter[step] + time_spent_w_counter[step]
            break


    f10_total_ += f10_total
    f10_b_ += f10_b
    f10_w_ += f10_w
    l10_total_ += l10_total
    l10_b_ += l10_b
    l10_w_ += l10_w

    f10_b_list.append(f10_b)
    f10_w_list.append(f10_w)
    l10_b_list.append(l10_b)
    l10_w_list.append(l10_w)
print('Figure 3 first w, first b, last w, last b', f10_w_/f10_total_, f10_b_/f10_total_, l10_w_/l10_total_, l10_b_/l10_total_)

print('Schmidt REF, NONREF, BOTH (in %)', SchmidtRESSummary/nParticipantsSchmidt*100)

list_for_anova = [f10_b_list, f10_w_list, l10_b_list, l10_w_list]
anova_Schmidt = u.anova(np.array(list_for_anova))
print('ANOVA Seijts Time + Discrepancy:', anova_Schmidt)

list_for_anova2 = [f10_b_list+l10_b_list, l10_w_list+f10_w_list]
anova_Schmidt = u.anova(np.array(list_for_anova2))
print('ANOVA Seijts Discrepancy:', anova_Schmidt)




#
#
#
# '''
# Seijts & Latham, 2001
# '''
#
# print('\nSeijts & Latham, 2001')
#
#
# statesSeijts = ['Outcome', 'Learning', 'DYB']
#
# statesVectorSeijts = u.encoder(statesSeijts)
#
# personTypesSeijts = ['A Group']
#
# nParticipantsSeijts = int((62+32)/3)
#
# groupSizesSeijts = [nParticipantsSeijts]
#
#
# drivesSeijts = ['Achievement', 'Honor', 'Curiosity']
#
# #                                      Ach  Hon Cur
# stateAndDriveToSatisfactionSeijts = [[[.7, -.4, .1],   # O
#                                       [.7, -.1, .6],   # L
#                                       [.7, -.0, .5]]]  # DYB
#
# #                                            Ach Hon
# SeijtsExperimentPersonToDeficitMapping = [.7, .6, .7]
# SeijtsExperimentPersonToDeficitMapping = u.normalDist(SeijtsExperimentPersonToDeficitMapping, nParticipantsSeijts, sigma=0.1)
#
#
# #                                Ach Hon Cur
# stateToStimulusMappingSeijts = [[.8, .3, .1],  # O
#                                 [.8, .1, .5],  # L
#                                 [.5, .0, .4]]  # DYB
#
#
# # #                                   Ach Hon Cur
# # driveAndStateToRelevanceSeijts = [[[.9, .7, .8],   # O
# #                                    [.9, .7, .8],   # L
# #                                    [.9, .7, .8]]]  # DYB
#
# SeijtsRES = u.getResults(id=0, groupSizes=groupSizesSeijts, states=statesVectorSeijts,
#                         driveAndStateToRelevance=None,#driveAndStateToRelevanceSeijts,
#                         stateAndDriveToSatisfaction=stateAndDriveToSatisfactionSeijts,
#                         stateToStimulusMapping=stateToStimulusMappingSeijts,
#                         deficits=SeijtsExperimentPersonToDeficitMapping, rules=rules, goals=['just do it'],
#                         effort=effort)
#
# score_Seijts, sd_Seijts, se_Seijts = u.calcPerformance(SeijtsRES, 1.825)
# print('Seijts M =', score_Seijts)
# print('Seijts SD =', sd_Seijts)
#
# anova_SeijtsOLDYB = u.anova(u.resToFreq(SeijtsRES))
# print('ANOVA Seijts Outcome vs Learning vs DYB:', anova_SeijtsOLDYB)
# print(u.t_test(states=statesSeijts, res=SeijtsRES))
#
# u.plot_results(personTypesSeijts, [score_Seijts], statesSeijts, 'Seijts')
#
#
#
#
#
#
#
#
# '''
# Chen & Latham, 2014
# M = 9.28, 7.09, 7.71, 6.69
# '''
#
# print('\nChen & Latham, 2014')
#
# statesChen = ['Learning', 'Performance', 'Learning & Performance', 'Control']
#
# statesVectorChen = u.encoder(statesChen)
#
# personTypesChen = ['A Group']
#
# nParticipantsChen = 22
#
# groupSizesChen = [nParticipantsChen]
#
#
# drivesChen = ['Curiosity', 'Honor']
#
# #                                    Cur  Hon
# stateAndDriveToSatisfactionChen = [[[1.,  .0],   # L
#                                     [1., -.55],  # P
#                                     [1., -.35],   # PL
#                                     [1.,  .0]]]  # C
#
# #                                       Cur Hon
# ChenExperimentPersonToDeficitMapping = [.7, .7]
# ChenExperimentPersonToDeficitMapping = u.normalDist(ChenExperimentPersonToDeficitMapping, nParticipantsChen)
#
#
# # #                              Cur  Hon
# # stateToStimulusMappingChen = [[1.,  .1],  # L
# #                               [.8,  .7],  # P
# #                               [.62, .4],  # PL
# #                               [.4,  .1]]  # C
#
# #                              Cur  Hon
# stateToStimulusMappingChen = [[1.,  .1],  # L
#                               [.8,  .7],  # P
#                               [.9, .4],  # PL
#                               [.4,  .1]]  # C
#
#
#
#
# #                                 Cur Hon
# # driveAndStateToRelevanceChen = [[[.9, .7],   # L
# #                                  [.9, .7],   # P
# #                                  [.9, .7],   # PL
# #                                  [.9, .7]]]  # C
#
# ChenRES = u.getResults(id=0, groupSizes=groupSizesChen, states=statesVectorChen,
#                         driveAndStateToRelevance=None,#driveAndStateToRelevanceChen,
#                         stateAndDriveToSatisfaction=stateAndDriveToSatisfactionChen,
#                         stateToStimulusMapping=stateToStimulusMappingChen,
#                         deficits=ChenExperimentPersonToDeficitMapping, rules=rules, goals=['just do it'],
#                         effort=effort)
#
# score_Chen, sd_Chen, se_Chen = u.calcPerformance(ChenRES, 1.93)
# print('Chen M =', score_Chen)
# print('Chen SD =', sd_Chen)
#
# anova_Chen = u.anova(u.resToFreq(ChenRES))
# print('ANOVA Chen Condition:', anova_Chen)
# print(u.t_test(states=statesChen, res=ChenRES))
#
# u.plot_results(personTypesChen, [score_Chen], statesChen, 'Chen')
#
#
#
#
#
#
# '''
# Cianci, Klein & Seijts (2010)
# M = 9.28, 7.09, 7.71, 6.69
# '''
#
# print('\nCianci, Klein & Seijts (2010)')
#
#
# statesCianci = ['Learning', 'Performance']
#
# statesVectorCianci = u.encoder(statesCianci)
#
# personTypesCianci = ['Low Conscientiousness Group', 'High Conscientiousness Group']
#
# nParticipantsCianci = 37
#
# groupSizesCianci = [nParticipantsCianci, nParticipantsCianci]
#
#
# drivesCianci = ['Achievement', 'Honor']
#
# #                                     Ach   Hon
# stateAndDriveToSatisfactionCianci = [[[1.,  .0],   # L
#                                       [1., -.8]]]  # P
#
# #                                     Ach Hon
# CianciLowConPersonToDeficitMapping = [.7, .6]
# CianciLowConPersonToDeficitMapping = u.normalDist(CianciLowConPersonToDeficitMapping, nParticipantsCianci)
#
# #                                      Ach Hon
# CianciHighConPersonToDeficitMapping = [.9, .9]
# CianciHighConPersonToDeficitMapping = u.normalDist(CianciHighConPersonToDeficitMapping, nParticipantsCianci)
#
#
# #                                Ach Hon
# stateToStimulusMappingCianci = [[1., .4],  # L
#                                 [.7, .9]]  # P
#
#
# # #                                   Ach Hon
# # driveAndStateToRelevanceCianci = [[[.9, .7],   # L
# #                                    [.9, .7]]]  # P
#
# CianciLowC_RES = u.getResults(id=0, groupSizes=groupSizesCianci, states=statesVectorCianci,
#                         driveAndStateToRelevance=None,#driveAndStateToRelevanceCianci,
#                         stateAndDriveToSatisfaction=stateAndDriveToSatisfactionCianci,
#                         stateToStimulusMapping=stateToStimulusMappingCianci,
#                         deficits=CianciLowConPersonToDeficitMapping, rules=rules, goals=['just do it'],
#                         effort=effort)
#
# CianciHighC_RES = u.getResults(id=0, groupSizes=groupSizesCianci, states=statesVectorCianci,
#                         driveAndStateToRelevance=None,#driveAndStateToRelevanceCianci,
#                         stateAndDriveToSatisfaction=stateAndDriveToSatisfactionCianci,
#                         stateToStimulusMapping=stateToStimulusMappingCianci,
#                         deficits=CianciHighConPersonToDeficitMapping, rules=rules, goals=['just do it'],
#                         effort=effort)
#
# score_CianciL, sd_Cianci, _ = u.calcPerformance(CianciLowC_RES, 2.3)
# print('Cianci Low C. M =', score_CianciL)
# print('Cianci Low C. SD =', sd_Cianci)
#
# score_CianciH, sd_Cianci, _ = u.calcPerformance(CianciHighC_RES, 2.3)
# print('Cianci High C. M =', score_CianciH)
# print('Cianci High C. SD =', sd_Cianci)
#
# anova_Cianci1 = u.anova(u.resToFreq(CianciLowC_RES+CianciHighC_RES))
# print('ANOVA Cianci Outcome vs Learning:', anova_Cianci1)
#
# anova_Cianci2 = u.anova(u.resToFreq([CianciLowC_RES[0]+CianciLowC_RES[1], CianciHighC_RES[0]+CianciHighC_RES[1]]))
# print('ANOVA Cianci Low Conscientiousness vs High Conscientiousness:', anova_Cianci2)
#
# u.plot_results(personTypesCianci, [score_CianciL, score_CianciH], statesCianci, 'Cianci')
#
#
#
#
#
#
# '''
# Shih & Alexander (2000)
# M = 9.28, 7.09, 7.71, 6.69
# '''
#
# print('\nShih & Alexander (2000)')
#
# statesShih = ['Goal_Self', 'Goal_Social', 'No_Goal_Self', 'No_Goal_Social']
#
# statesVectorShih = u.encoder(statesShih)
#
# personTypesShih = ['A Group']
#
# nParticipantsShih = 42*2
#
# groupSizesShih = [nParticipantsShih]
#
# drivesShih = ['Achievement', 'Honor', 'Avoid Unpleasantness']
#
# #                                    Ach  Hon  AUn
# stateAndDriveToSatisfactionShih = [[[.7, -.1,  .0],   # GSe
#                                     [.7, -.2, -.2],   # GSo
#                                     [.7, -.1,  .0],   # NGSe
#                                     [.7, -.2, -.2]]]  # NGSo
#
# #                                       Ach Hon AUn
# ShihExperimentPersonToDeficitMapping = [.7, .7, .7]
# ShihExperimentPersonToDeficitMapping = u.normalDist(ShihExperimentPersonToDeficitMapping, nParticipantsShih, sigma=.01)
#
#
# #                              Ach Hon  AUn
# stateToStimulusMappingShih = [[.7, .1,  .1],   # GSe
#                               [.7, .3,  .2],   # GSo
#                               [.7, .1,  .1],   # NGSe
#                               [.7, .3,  .4]]   # NGSo
#
# ShihRES = u.getResults(id=0, groupSizes=groupSizesShih, states=statesVectorShih,
#                         driveAndStateToRelevance=None,
#                         stateAndDriveToSatisfaction=stateAndDriveToSatisfactionShih,
#                         stateToStimulusMapping=stateToStimulusMappingShih,
#                         deficits=ShihExperimentPersonToDeficitMapping, rules=rules, goals=['just do it'],
#                         effort=effort)
#
#
# score_Shih, sd_Shih, se_Shih = u.calcPerformance(ShihRES, 15)
# print('Shih M =', score_Shih)
# print('Shih SD =', sd_Shih)
#
# anova_Shih = u.anova(u.resToFreq(ShihRES))
# print('ANOVA Shih Goal Self vs Goal Social vs No Goal Self vs No Goal Social:', anova_Shih)
# print(u.t_test(states=statesShih, res=ShihRES))
#
# ShihRES_GvsNG = np.array([np.sum(ShihRES[:2], axis=0), np.sum(ShihRES[2:], axis=0)])
#
# anova_Shih_GvsNG = u.anova(u.resToFreq(ShihRES_GvsNG))
# print('ANOVA Shih Goal vs No Goal:', anova_Shih_GvsNG)
#
# u.plot_results(groupSizesShih, [score_Shih], statesShih, 'Shih')
#
#
#
