import cv2
import numpy as np

momentsWithContours = []
namesWithMoments = []

value = 331
debug = False
tolerance = 0.08


def loadTemplates():
    files = ['dydolce', 'kolo', 'parowki', 'audi', 'waskie']
    path = 'images/wzor/'
    extension = '.jpg'
    svalue = 151
    for file in files:
        image = cv2.imread(path + file + extension)
        cards = getAllCards(image)
        internalCard = cutInternalCard(cards[0][1])
        moments = getAllHuMoments(internalCard)
        namesWithMoments.append((file, moments))
        # print(file)
        # print(moments)
        # plt.subplot(svalue)
        # plt.xticks([])
        # plt.yticks([])
        # plt.imshow(internalCard)
        # svalue += 1
    print()


def getCardName(cardMoments):
    for (name, moments) in namesWithMoments:
        if theSame(cardMoments, moments):
            return name
    return "nieznany ksztalt"


def getAllHuMoments(image):
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 180, 255, 0)
        a = cv2.HuMoments(cv2.moments(image)).flatten()
        # plt.subplot(value)
        # plt.xticks([])
        # plt.yticks([])
        # plt.imshow(image)
        global value
        value += 1
        if a.all() == 0:
            return []
        # return a
        return np.log10(np.abs(a))
    return []


def getContoursToDraw():
    toReturn = []
    for i, this in enumerate(momentsWithContours):
        for j, other in enumerate(momentsWithContours):
            if i == j:
                continue
            if theSame(this[0], other[0]):
                toReturn.append(this[1])
                toReturn.append(other[1])
    return toReturn


def isCard(contours):
    cardPattern = cv2.imread("images/wzor/karta.jpg")
    cardPattern = cv2.cvtColor(cardPattern, cv2.COLOR_BGR2GRAY)
    _, cardThresh, = cv2.threshold(cardPattern, 100, 255, 0)
    _, patternContour, _ = cv2.findContours(cardPattern, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ret = cv2.matchShapes(patternContour[0], contours, 1, 0.0)
    if ret > 0.5:
        return False
    else:
        return True


def theSame(one, two):
    # for a, b in zip(one[0], two[0]):
    x = abs(one[0]-two[0])
    # x = 0
    # for i in range(0, 7):
    #     x += abs((1/one[i]) - (1/two[i]))
    y = tolerance
    if x > y:
        return False
    return True


def countNotBlackPixels(imageWithMask):
    count = 0
    for pixel in imageWithMask:
        if pixel.any() > 0:
            count += 1
    return count


def detectColor(image):
    maxLitPixels = (0, "key", 0)
    allBoundaries = {
        "yellow": ([163, 74, 3], [238, 169, 75]),
        "blue": ([40, 60, 99], [149, 134, 134]),
        "red": ([130, 5, 5], [255, 84, 69]),
        "green": ([5, 75, 20], [77, 134, 83])
    }

    for key, (lower, upper) in allBoundaries.items():
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        litPixels = countNotBlackPixels(output)
        if litPixels > maxLitPixels[2]:
            maxLitPixels = (output, key, litPixels)
    return maxLitPixels


def adjustGamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def cropMinAreaRect(img, rect):
    expander = 7
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    img_crop = img_rot[pts[1][1] - expander:pts[0][1] + expander,
               pts[1][0] - expander:pts[2][0] + expander]

    return img_crop


def getAllCards(image):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 140, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    allCards = []
    for contour in contours:
        if isCard(contour):
            rect = cv2.minAreaRect(contour)
            croppedCard = cropMinAreaRect(image, rect)
            height, width, channels = croppedCard.shape
            if height > 50 and width > 50:
                allCards.append((rect, croppedCard, cv2.boundingRect(contour)))
    return allCards


def cutInternalCard(card):
    height, width, channels = card.shape
    cardArea = height * width
    card = adjustGamma(card, 1.2)
    cardGray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    _, cardThresh = cv2.threshold(cardGray, 180, 255, 0)
    _, cardContours, _ = cv2.findContours(cardThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cardContours:
        areaRatio = cv2.contourArea(cnt) / cardArea
        if 0.05 < areaRatio < 0.5:
            rect = cv2.minAreaRect(cnt)
            result = cropMinAreaRect(card, rect)
            return result


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def drawDebugInfo(moments, originalImg, x, y, w, h):
    cv2.rectangle(originalImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
    y0, dy = y, 20
    for i, line in enumerate(moments):
        y = y0 + i * dy
        cv2.putText(originalImg, str(line), (x - 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


def main():
    picture = "match5_2"
    loadTemplates()
    originalImg = cv2.imread("images/" + picture + '.jpeg')
    imageBig = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
    cards = getAllCards(imageBig)
    print("Wykryto %s kart(y)" % len(cards))
    subplotValue = 100 + (10 * len(cards)) + 10 + 1
    for rect, card, (x, y, w, h) in cards:
        try:

            internalCard = cutInternalCard(card)
            moments = getAllHuMoments(internalCard)

            if len(moments) == 0:
                continue
            print(moments)

            momentsWithContours.append([moments, (x, y, w, h)])
            if debug:
                drawDebugInfo(moments, originalImg, x, y, w, h)

            name = getCardName(moments)
            cv2.putText(originalImg, name, (x + w - 100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

            # plt.subplot(subplotValue)
            # plt.xticks([])
            # plt.yticks([])
            # plt.imshow(card)
            # subplotValue += 1
        except ValueError:
            pass

    # plt.subplot(subplotValue)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(imageBig)
    # plt.savefig("images/results/cut3.png")
    # plt.show()

    contoursToDraw = getContoursToDraw()
    for (x, y, w, h) in contoursToDraw:
        cv2.rectangle(originalImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("images/results/" + picture + 'result.jpg', image_resize(originalImg, height=700))


main()
