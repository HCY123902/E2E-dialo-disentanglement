#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import sys

import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a conversation graph to a set of connected components (i.e. threads).')
    parser.add_argument('--raw_list', help='Path to the text file that contains the list of paths of raw text documents in a particular split')
    parser.add_argument('--cluster', help='Path to the file containing the cluster content of a particular split')
    parser.add_argument('--result', help='Path of the resulting json')

    args = parser.parse_args()

    clusters = {}
    for line in open(args.cluster):
        time = ''
        if ':' in line:
            time, line = line.split(":")
        nums = [int(v) for v in line.strip().split()]
        nums.sort()
        clusters.setdefault(time, []).append(nums)

    text = {}
    for file_list in open(args.raw_list):
        filename = file_list.split(' ')[0]
        print("current raw file is {}".format(filename))
        time = (filename.split("/")[-1]).split(".")[0]
        for line in open(filename):
            # print("parsed time is {}".format(time))
            if line[0] == '[':
                parts = line.split("]")
                #filename = parts[0]
                line = ''.join(parts[1:])
            text.setdefault(time, []).append(line.strip())

    result_path = "{}.json".format(args.result)
    result = open(result_path, "w")

    dialogues = []
    smallest_cluster_number = 1000
    largest_cluster_number = 0
    average_cluster_number = 0

    for time in clusters:
        sortable_clusters = []
        for cluster in clusters[time]:
            size = len(cluster)
            first = min(cluster)
            sortable_clusters.append((first, size, cluster))
        sortable_clusters.sort()

        turn_to_cluster_map = {}
        number_of_turns = len(text[time])
        for i, _, cluster in sortable_clusters:
            for turn_position in cluster:
                turn_to_cluster_map[turn_position] = i


        for i in range(1000, number_of_turns, 50):
            number_of_sample_turns = min(50, number_of_turns - i)
            start_cluster = min([turn_to_cluster_map[turn_position] for turn_position in range(i, i + number_of_sample_turns)])
            sample = []

            cluster_map = {}
            for j in range(i, i + number_of_sample_turns):
                turn_text = text[time][j]

                if turn_text[0] != "<":
                    print(turn_text)
                    continue
                
                if turn_text[:3] == "===":
                    label = turn_to_cluster_map[j] - start_cluster
                    sample.append({"speaker": "system", "utterance": turn_text[3:], "label": label})
                else:
                    speaker_turn_text = turn_text.split(">", 1)
                    speaker = speaker_turn_text[0][1:]

                    turn_text = speaker_turn_text[1].strip()
                    label = turn_to_cluster_map[j] - start_cluster
                    sample.append({"speaker": speaker, "utterance": turn_text, "label": label})

                if turn_to_cluster_map[j] not in cluster_map:
                    cluster_map[turn_to_cluster_map[j]] = 1

            if len(cluster_map) > 15:
                print("There is a sample with {} clusters containing messages at time {} from {} to {}".format(len(cluster_map), time, i, i + number_of_sample_turns))
            if len(cluster_map) < smallest_cluster_number:
                smallest_cluster_number = len(cluster_map)
            if len(cluster_map) > largest_cluster_number:
                largest_cluster_number = len(cluster_map)
            average_cluster_number = average_cluster_number + len(cluster_map)

            dialogues.append(sample)
    
    average_cluster_number = average_cluster_number / len(dialogues)
    
    print("The number of samples is {}".format(len(dialogues)))
    print("The smallest cluster number is {}".format(smallest_cluster_number))
    print("The largest cluster number is {}".format(largest_cluster_number))
    print("The average cluster number is {}".format(average_cluster_number))

    json.dump(dialogues, result)
