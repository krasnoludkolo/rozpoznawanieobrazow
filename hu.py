import cv2
import matplotlib.pyplot as plt
import numpy as np

momentsWithContours = []


def getAllHuMoments(image):
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        a = cv2.HuMoments(cv2.moments(image)).flatten()
        if a.all() == 0:
            return []
        # return a
        return -np.sign(a) * np.log10(np.abs(a))
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


def theSame(one, two):
    tolerance = 1
    for a, b in zip(one[:-1], two[:-1]):
        x = abs(a-b)
        y = tolerance
        if x > y:
            return False
    return True


def cropMinAreaRect(img, rect):
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]

    return img_crop


def getAllCards(image):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 140, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    allCards = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        croppedCard = cropMinAreaRect(image, rect)
        allCards.append((rect, croppedCard,cv2.boundingRect(contour)))
    return allCards


def main():
    file = "images/2tr.jpeg"
    im = cv2.imread(file)
    subplotValue = 141
    for rect, card, (x,y,w,h) in getAllCards(im):

        if w < 10 or h < 10:
            continue

        moments = getAllHuMoments(card)

        if len(moments) == 0:
            continue
        print(moments)
        print()

        momentsWithContours.append([moments, (x, y, w, h)])
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 100, 100), 2)

        y0, dy = y, 20
        for i, line in enumerate(moments):
            y = y0 + i * dy
            cv2.putText(im, str(line), (x-100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        try:
            plt.subplot(subplotValue)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(card)
            subplotValue += 1
        except ValueError:
            pass

    contoursToDraw = getContoursToDraw()
    for (x, y, w, h) in contoursToDraw:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)


    plt.subplot(subplotValue)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im)
    plt.savefig("kolory4.png")
    plt.show()
    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
