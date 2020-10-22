import cv2
import matplotlib
import numpy as np
import math
from matplotlib import pyplot as plt
import imutils
from graphManager import Graph
import scipy
import networkx
import difflib


def borders(x, y, width, height):
    isNorth = False
    isSouth = False
    isEast = False
    isWest = False
    borderData = []
    # The values of 7, 260, and 480 were chose to allow for a building to count as on a border when there is whitespace
    # separating it from the actual border. It was noted that there was more whitespace on the southern and eastern
    # border, so their margins are larger.
    if x < 7:
        isWest = True
    if x + width > 260:
        isEast = True
    if y < 7:
        isNorth = True
    elif y + height > 480:
        isSouth = True

    if not isNorth and not isSouth:
        borderData.append("NotNorternmostNorSouthernmost")
    elif isNorth:
        borderData.append("onNorthBorder")
    else:
        borderData.append("onSouthBorder")

    if isEast or isWest:
        if isEast:
            borderData.append("onEastBorder")
        if isWest:
            borderData.append("onWestBorder")
    else:
        borderData.append("NotEasternmostNorWesternmost")

    return borderData


def orientation(cnt, img):
    try:
        ellipse = cv2.fitEllipse(cnt)
        im = cv2.ellipse(img, ellipse, (0, 255, 0), 2)
        # print(ellipse[2])

        #Due to the fact that the orientation of the building is perpendicular to that of the ellipse surrounding it,
        # within 1 degree of 90 means it is perfectly flat, and within 2 degrees of 180 or 0 means it is horizontal

        if 91 > ellipse[2] > 89:
            (x, y, w, h) = cv2.boundingRect(cnt)
            ar = w / float(h)

            if 0.95 <= ar <= 1.05:
                return "NoSingleOrientation"
            else:
                return "orientedEast2West"
        elif 178 < ellipse[2] < 182 or ellipse[2] < 2:
            return "orientedNorth2South"
        else:
            return "noSingleOrientation"

            # return ellipse
    except:
        # print("ellipse failed")
        return "orientedEast2West"

    # return "orientedEast2West"


def isSymmetric(c, cX, cY):
    labeled = labeledMap
    labeledImage = cv2.imread("new-labeled.pgm")
    value = labeled[cY][cX]
    y = cY
    if value == 0:
        while value == 0:
            y -= 1
            value = labeled[y][cX]
    upper = np.array([value + 1, value + 1, value + 1])
    lower = np.array([value - 1, value - 1, value - 1])
    # print("Value: ", value)
    mask = cv2.inRange(labeledImage, lower, upper)
    # cv2.imshow("mask", mask)
    flipped = cv2.flip(mask, 1)
    # cv2.imshow("flipped", flipped)
    cnts, hierarchy = cv2.findContours(flipped.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # print(len(cnts))
    for item in cnts:
        ret = cv2.matchShapes(item, c, 1, 0.0)
        # print("Ret: ", ret)
    # Ret was chosen to be < 0.5 because if two items are less tan 50% different when reflected and compared, then they
    # are likely symmetric. Ret >3 was also included because for some reason, one building that was visually symmetrical
    # was deemed highly different, so this accounts for that strange outlier.
    if ret < 0.5 or ret > 3:
        return "isSymmetric", value
    else:
        return "isNotSymmetric", value


def complexShapes(c, ratio):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    if len(approx) < 4:
        shape = "unknown"
        return shape
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangular"
        return shape
    else:
        x, y, w, h = cv2.boundingRect(c)
        x *= ratio
        y *= ratio
        w *= ratio
        h *= ratio
        w = int(w)
        h = int(h)
        # print(w, ", ", h)
        shapeArray = np.zeros((h, w))
        currY = int(y)
        # print("shapeArray: ", shapeArray)
        # print(shapeArray.shape)
        for r in range(h):
            # print("r: ", r)
            currX = int(x)
            for c in range(w):
                # print()
                # print(currX,",", currY)
                shapeArray[r][c] = labeledMap[currY][currX]
                currX += 1
            currY += 1
        if isLShaped(shapeArray):
            return "LShaped"

        if isCShaped(shapeArray):
            return "CShaped"

        if isIShaped(shapeArray):
            return "IShaped"
        else:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "complicatedSquare" if 0.95 <= ar <= 1.05 else "complicatedRectangle"
            return shape


def isCShaped(arr):
    if isVerticalCShaped(arr) or isHorizontalCShaped(arr):
        return True
    return False


def isHorizontalCShaped(arr):
    blocks = 0
    currentBlocks = 0
    check = True
    for r in range(len(arr)):
        if arr[r][0] != 0:
            check = False
            currentBlocks += 1
        for c in range(len(arr[0])):
            if arr[r][c] == 0:
                check = True
            else:
                if check:
                    check = False
                    currentBlocks += 1
        if currentBlocks == 2:
            blocks += 1
        currentBlocks = 0
    if len(arr) > blocks >= int(len(arr) * 1 / 3 - 1):
        # This value was chosen because it accurately distinguished between a horizontal C and other shapes that are
        # visibly not c-shaped
        return True
    return False


def isVerticalCShaped(arr):
    np.delete(arr, 0, axis=0)
    blocks = 0
    currentBlocks = 0
    check = True
    for c in range(len(arr[0])):
        if arr[0][c] != 0:
            check = False
            currentBlocks += 1
        for r in range(len(arr)):
            if arr[r][c] == 0:
                check = True
            else:
                if check:
                    check = False
                    currentBlocks += 1
        if currentBlocks == 2:
            blocks += 1
        currentBlocks = 0
    if len(arr[0]) > blocks > len(arr[0]) * 2 / 3:
        # This value was chosen because it was able to differentiate between shapes that were visibly C shaped and those
        # that were visibly I-shaped
        return True
    return False


def isIShaped(arr):
    np.delete(arr, 0, axis=0)
    blocks = 0
    currentBlocks = 0
    check = True
    for c in range(len(arr[0])):
        if arr[0][c] != 0:
            check = False
            currentBlocks += 1
        for r in range(len(arr)):
            if arr[r][c] == 0:
                check = True
            else:
                if check:
                    check = False
                    currentBlocks += 1
        if currentBlocks == 2:
            blocks += 1
        currentBlocks = 0
    if len(arr[0]) > blocks >= int(len(arr[0]) * 1 / 3):
        # This value was chosen because it was able to differentiate between I shapes and simply rectangles with shapes
        # cut out of them
        return True
    return False
    pass


def isLShaped(arr):
    np.delete(arr, 0, axis=0)
    blocks = 0
    currentBlocks = 0
    check = True
    for c in range(len(arr[0])):
        if arr[0][c] != 0:
            check = False
            currentBlocks += 1
        for r in range(len(arr)):
            if arr[r][c] == 0:
                check = True
            else:
                if check:
                    check = False
                    currentBlocks += 1
        if currentBlocks < 2:
            blocks += 1
        currentBlocks = 0
    if len(arr[0]) > blocks > len(arr[0]) * 17 / 24:
        # This value was chosen because it was able to differentiate between L shapes and other shapes, even when the L
        # shape has extra blocks extruding from it's core shape.
        return True
    return False


def getRelations(shapes):
    north = []
    south = []
    east = []
    west = []
    near = []
    relations = [north, south, east, west, near]
    for source in shapes:
        n = []
        s = []
        e = []
        w = []
        nr = []
        for target in shapes:
            src = source[0]
            t = target[0]
            n.append(North(src, t))
            s.append(South(src, t))
            e.append(East(src, t))
            w.append(West(src, t))
            nr.append(Near(src, t))
        north.append(n)
        south.append(s)
        east.append(e)
        west.append(w)
        near.append(nr)

    for r in range(26):
        replaceNorth = False
        replaceSouth = False
        for c in range(26):
            if replaceNorth:
                north[r][c] = 0
            elif north[r][c] == 1:
                replaceNorth = True
            if replaceSouth:
                south[r][25 - c] = 0
            elif south[r][25 - c] == 1:
                replaceSouth = True
            if east[r][c] != 0:
                east[r] = removeDuplicates(east[r], east[c], c, True)
            if west[r][c] != 0:
                west[r] = removeDuplicates(west[r], west[c], c, False)

    return relations


def removeDuplicates(currRow, row, i, east):
    if east:
        for x in range(i, len(currRow)):
            if currRow[x] == row[x]:
                currRow[x] = 0
        return currRow
    else:
        index = i
        while index > 0:
            if currRow[index] == row[index]:
                currRow[index] = 0
            index -= 1
        return currRow


def North(s, t):
    north = None
    img = cntImage.copy()
    Ms = cv2.moments(s)
    Mt = cv2.moments(t)
    csx = int((Ms["m10"] / Ms["m00"]))
    csy = int((Ms["m01"] / Ms["m00"]))
    ctx = int((Mt["m10"] / Mt["m00"]))
    cty = int((Mt["m01"] / Mt["m00"]))
    sx, sy, sw, sh = cv2.boundingRect(s)
    tx, ty, tw, th = cv2.boundingRect(t)

    if csy > ty + th:
        north = 1

    else:
        north = 0

    return north


def South(s, t):
    south = None
    img = cntImage.copy()
    Ms = cv2.moments(s)
    Mt = cv2.moments(t)
    csx = int((Ms["m10"] / Ms["m00"]))
    csy = int((Ms["m01"] / Ms["m00"]))
    ctx = int((Mt["m10"] / Mt["m00"]))
    cty = int((Mt["m01"] / Mt["m00"]))
    sx, sy, sw, sh = cv2.boundingRect(s)
    tx, ty, tw, th = cv2.boundingRect(t)

    if ty > sy + sh:
        south = 1

    else:
        south = 0

    return south


def East(s, t):
    east = None
    img = cntImage.copy()
    Ms = cv2.moments(s)
    Mt = cv2.moments(t)
    csx = int((Ms["m10"] / Ms["m00"]))
    csy = int((Ms["m01"] / Ms["m00"]))
    ctx = int((Mt["m10"] / Mt["m00"]))
    cty = int((Mt["m01"] / Mt["m00"]))
    sx, sy, sw, sh = cv2.boundingRect(s)
    tx, ty, tw, th = cv2.boundingRect(t)

    if sx + sw < tx:
        east = 1

    else:
        east = 0
    return east


def West(s, t):
    west = None
    img = cntImage.copy()
    Ms = cv2.moments(s)
    Mt = cv2.moments(t)
    csx = int((Ms["m10"] / Ms["m00"]))
    csy = int((Ms["m01"] / Ms["m00"]))
    ctx = int((Mt["m10"] / Mt["m00"]))
    cty = int((Mt["m01"] / Mt["m00"]))
    sx, sy, sw, sh = cv2.boundingRect(s)
    tx, ty, tw, th = cv2.boundingRect(t)
    if sx > tx + tw:
        west = 1
    else:
        west = 0
    return west


MAX_DIST = 160
# This value was selected by visually comparing distance returns from the Near(s,t) function and selecting a value that
# properly restricted buildings that were near each other, based on input from a group of 4 other people

def Near(s, t):
    img = cntImage.copy()
    Ms = cv2.moments(s)
    Mt = cv2.moments(t)
    csx = int((Ms["m10"] / Ms["m00"]))
    csy = int((Ms["m01"] / Ms["m00"]))
    ctx = int((Mt["m10"] / Mt["m00"]))
    cty = int((Mt["m01"] / Mt["m00"]))
    sx, sy, sw, sh = cv2.boundingRect(s)
    tx, ty, tw, th = cv2.boundingRect(t)

    minDist = float("inf")

    sPoints = [(sx, sy), (csx, sy), (sx + sw, sy),
               (sx, csy), (csx, csy), (sx + sw, csy),
               (sx, sy + sh), (csx, sy + sh), (sx + sw, sy + sh)]
    tPoints = [(tx, ty), (ctx, ty), (tx + tw, ty),
               (tx, cty), (ctx, cty), (tx + tw, cty),
               (tx, ty + th), (ctx, ty + th), (tx + tw, ty + th)]

    for p1 in sPoints:
        for p2 in tPoints:
            dist = distance(p1, p2)
            if dist < minDist:
                minDist = dist

    if minDist <= MAX_DIST and minDist != 0:
        return 1
    return 0


def printWhats(data):
    # print(data)
    whats = []
    for item in data:
        row = []
        row.append(item[1])
        row.append(item[10])
        row.append(item[11])
        row.append(item[12])
        row.append(item[13])
        row.append(item[2])
        row.append(item[5])
        row.append(item[6])
        row.append(item[9])
        whats.append(row)

    return whats

def printWhere(data):
    print(data)
    whats = printWhats(data)

    for i in range(len(whats)):
        whats[i].append(data[i][8])
        whats[i].append(data[i][7])
        whats[i].append(data[i][4])
        whats[i].append(data[i][3])
        print(whats[i])


def distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


cntImage = None

ratio = 0


def findShapes():
    global ratio
    global cntImage
    image = cv2.imread("new-campus.pgm")
    im2 = cv2.imread("new-campus.pgm")
    resized = imutils.resize(image, width=1000)
    ratio = image.shape[0] / float(resized.shape[0])
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    index = 0
    cntImage = resized.copy()
    overallData = []
    skip = False
    for c in cnts:
        if skip:
            skip = False
            index += 1
        else:
            shapeData = []
            shapeData.insert(0, c)
            M = cv2.moments(c)
            area = cv2.contourArea(c) * (ratio * ratio)
            if area < 1000:
                shapeData.append("small")
            elif area < 3000:
                shapeData.append("medium")
            else:
                shapeData.append("large")
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            size = image.shape

            if cY < size[0]*3 / 8:
                shapeData.append("NorthHalf")
            elif cY > size[0]*5/8:
                shapeData.append("SouthHalf")
            else:
                shapeData.append("verticallyCentered")
            if cX < size[1] / 3:
                shapeData.append("WestHalf")
            elif cX > size[1]*2/3:
                shapeData.append("EastHalf")
            else:
                shapeData.append("horizontallyCentered")
            shape = complexShapes(c, ratio)
            shapeData.append(shape)
            symmetric = isSymmetric(c, cX, cY)
            shapeData.append(symmetric[0])
            value = symmetric[1]
            for item in placeNames:
                if item[0] == value:
                    shapeData.insert(1, item[1])

            x, y, w, h = cv2.boundingRect(c)
            x *= ratio
            y *= ratio
            w *= ratio
            h *= ratio

            shapeData.append(borders(x, y, w, h))
            shapeData.append(orientation(c, resized))

            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            if hierarchy[0][index][3] == -1:
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

                cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
            if index < len(cnts) - 1:
                if hierarchy[0][index + 1][3] != -1:
                    # cv2.putText(image, "hole", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    #            0.5, (255, 0, 255), 2)
                    shapeData.append("hasHoles")
                    skip = True

                else:
                    shapeData.append("hasNoHoles")
            else:
                shapeData.append("hasNoHoles")

            shapeData.append((cX, cY))
            shapeData.append(area)
            shapeData.append((x, y))
            shapeData.append((x + w, y + h))
            # shapeData.insert(0, c)

            overallData.append(shapeData)
            # show the output image
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            index += 1

    return overallData


def readPGM(filepath):
    pgmf = open(filepath, "rb")
    header = pgmf.readline()
    (width, height) = [int(i) for i in header.split()[1:3]]
    # print(header)
    values = []
    image = cv2.imread(filepath)
    for item in image:
        # print("newItem")
        row = []
        # print(item)
        for i in item:
            row.append(i[1])
        values.append(row)
        # row.append()

    # for item in values:
    #   print(item)
    return values


labeledMap = readPGM("new-labeled.pgm")

placeNames = []


def getConfusability(item, otherItem):
    for i in range(2, len(item)):
        if type(item[i]) == type([1]):
            item[i] = tuple(item[i])
    # print(item[2:])
    for i in range(2, len(otherItem)):
        if type(otherItem[i]) == type([1]):
            otherItem[i] = tuple(otherItem[i])
    #    print(type(otherItem[i]))
    # print("Confusability: ", sm.ratio())

    res = len(set(item[2:]) & set(otherItem[2:])) / float(len(set(item[2:]) | set(otherItem[2:]))) * 100
    return res


def printDirections(G, path):
    # vertices = G.getVertices()
    start = G.getVertex(path[0])
    end = G.getVertex(path[len(path) - 1])
    print("Begin at ", start.name, ", which is the ", getDescription(start.data))
    print("End at ", end.name, ", which is the ", getDescription(end.data))
    for i in range(len(path) - 1):
        direction = []
        curr = G.getVertex(path[i]).data
        next = G.getVertex(path[i + 1]).data
        direction.append(North(curr[0], next[0]))
        direction.append(South(curr[0], next[0]))
        direction.append(East(curr[0], next[0]))
        direction.append(West(curr[0], next[0]))
        instruction = "Head "
        if direction[0] and direction[2]:
            instruction += " northeast "
        elif direction[0] and direction[3]:
            instruction += " northwest "
        elif direction[1] and direction[2]:
            instruction += " southeast "
        elif direction[1] and direction[3]:
            instruction += " southwest "
        elif direction[0]:
            instruction += " north "
        elif direction[1]:
            instruction += " south "
        elif direction[2]:
            instruction += " east "
        elif direction[3]:
            instruction += " west "

        instruction += "to the "
        instruction += getDescription(next)
        print(instruction)
    print("You have arrived at your destination")


def getDescription(building):
    #print(building)
    description = ""
    description += building[2]
    description += " "

    if building[5] != "rectangular" or building[5] != "square":
        if 'isSymmetric' in building:
            description += " symmetric "
        else:
            description += " not symmetric "
    description += building[5]
    description += " building "

    if building[8] == "noSingleOrientation":
        description += " that has no single orientation "
    else:
        description += " that is "
        description += building[8]
    # print(description)
    if len(building[7]) > 1:
        borders = building[7]
        if 'onNorthBorder' in borders:
            if "onWestBorder" in borders:
                description += " in the North-west corner "
            elif "onEastBorder" in borders:
                description += " in the North-east corner "
            elif 'NotEasternmostNorWesternmost' in borders:
                description += " on the northern border "
        elif 'onSouthBorder' in borders:
            if "onWestBorder" in borders:
                description += " in the South-west corner "
            elif "onEastBorder" in borders:
                description += " in the South-east corner "
            elif 'NotEasternmostNorWesternmost' in borders:
                description += " on the southern border "
        elif "NotNorternmostNorSouthernmost" in borders:
            if "onWestBorder" in borders:
                description += " on the western border "
            elif "onEastBorder" in borders:
                description += " on the eastern border"
            if building[3] == "NorthHalf":
                description += " in the northern half "
            if building[3] == "SouthHalf":
                description += " in the southern half "
    else:
        if 'onNorthBorder' in building[7]:
            description += " on the northern border"
        if "onWestBorder" in building[7]:
            description += " on the western border "
        if "onEastBorder" in building[7]:
            description += " on the eastern border"
        if 'onSouthBorder' in building[7]:
            description += " on the southern border"

    description += "."
    return description


def findPaths(data, names):
    relations = getRelations(data)
    G = Graph()
    direction = ["north", "south", "east", "west", "near"]
    index = 0

    combinedRelations = np.add(relations[0], relations[1])
    combinedRelations = np.add(combinedRelations, relations[2])
    combinedRelations = np.add(combinedRelations, relations[3])
    combinedRelations = np.add(combinedRelations, relations[4])

    for r in range(26):
        for c in range(26):
            if combinedRelations[r][c] > 1:
                combinedRelations[r][c] = 1
    for r in range(26):
        for c in range(26):
            if combinedRelations[r][c] == 1:
                v1 = G.addVertex(data[r][1])
                v2 = G.addVertex(data[c][1])
                v1.addData(data[r])
                v2.addData(data[c])
                cost = getConfusability(data[r], data[c])
                if relations[4][r][c] == 0:
                    cost += 60
                G.addEdge(v1.name, v2.name, cost)

    # for v in G:
    #    for w in v.getConnections():
    #        vid = v.getName()
    #        wid = w.getName()
    #        print("( ", vid, ", ", wid, ", ", v.getWeight(w), ")")

    G.shortestPath("Mathematics", "Philosophy")
    target = G.getVertex('Philosophy')
    path = [target.getName()]
    G.shortest(target, path)
    shortestPath = path[::-1]
    print('The shortest path : %s' % shortestPath)

    for i in range(len(shortestPath) - 1):
        s = shortestPath[i]
        #    print("source: ", s)
        t = shortestPath[i + 1]
        #    print("target: ", t)
        s = names.index(s)
        t = names.index(t)
        #    print("S:" ,s)
        #    print("T:" ,t)
        index = 0

        # print(combinedRelations[s][t])

    printDirections(G, shortestPath)


def main():
    global placeNames
    f = open("new-table.txt")
    line = f.readline()
    while line:
        row = line.split()
        row[0] = int(row[0])
        placeNames.append(row)
        line = f.readline()
    # print(placeNames)
    data = findShapes()
    names = []
    for item in data:
        names.append(item[1])

    #printWhats(data)
    #printWhere(data)
    #print(getConfusability(data[names.index("Lewisohn")][2:], data[names.index("Mathematics")][2:]))
    findPaths(data, names)
    #directions = ["north", "south", "east", "west", "near"]
    #index = 0
    #relations = getRelations(data)
    #for item in relations:

    #    for r in range(len(item)):
    #        relation = ""
    #        targets = []
    #        for c in range(len(item[r])):
    #            if item[r][c] == 1:
    #                targets.append(names[c])
    #        if len(targets) == 0:
    #            continue
    #        for place in targets:
    #            relation += place
    #            if len(targets) > 1 and place != targets[len(targets)-1]:
    #                relation += " and "
    #        relation += " "
    #        relation += directions[index]
    #        relation += " "
    #        relation += names[r]
    #        print(relation)
    #    index += 1



if __name__ == "__main__":
    main()
