'''
Converts FCC radio tower text files to csv format
'''

import csv
import math

def absoluteCoordinates(direction, degrees, minutes, seconds):
    abs_coords = degrees + minutes/60 + seconds/3600
    if (direction == 'S' or direction == 'W'):
        abs_coords *= -1
    return abs_coords

def max_distance(power, height): # in miles
    power = power.replace(' ', '')
    power = power.replace('kW', '')
    power = power.replace('-', '.5')
    if(float(height) < 0):
        height = -1* float(height)
    return ((3.57)*(math.sqrt(float(height)) + 1.22474487)* math.sqrt(math.sqrt(float(power))))

allLines = []

with open("disgusting_awful_data.txt", 'r') as inFile:
    for line in inFile:
        curLine = line.split('|')
        if (curLine[15].strip() == '-' or curLine[16].strip() == '-' or curLine[1].strip() == '-' or curLine[19].strip() == '-' or curLine[20].strip() == '-' or curLine[21].strip() == '-' or curLine[22].strip() == '-' or curLine[23].strip() == '-' or curLine[24].strip() == '-' or curLine[25].strip() == '-' or curLine[26].strip() == '-'):
            print("invalid location")
        else:
            freq = curLine[1].strip()
            power = curLine[15].strip()
            height = curLine[16].strip()
            absLat = absoluteCoordinates(curLine[19].strip(), float(curLine[20].strip()), float(curLine[21].strip()), float(curLine[22].strip()) )
            absLng = absoluteCoordinates(curLine[23].strip(), float(curLine[24].strip()), float(curLine[25].strip()), float(curLine[26].strip()) )

        cleanedLine = [curLine[1], absLat, absLng, max_distance(power, height)]
        cleanedLine[0] = cleanedLine[0].strip()
        allLines.append(cleanedLine)

with open('US_data_With_freq.csv', 'w', newline='') as outFile:
    writer = csv.writer(outFile)
    writer.writerows(allLines)



        
