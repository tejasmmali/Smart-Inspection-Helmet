import cv2

image = cv2.imread("gauge.jpg")


cv2.imshow("Gauge", image)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gauge (Grayscale)", gray)
print(image.shape)
print(gray.shape)

_, binary = cv2.threshold(
    gray, 100, 255, cv2.THRESH_BINARY
)
cv2.imshow("Gauge (Binary)", binary)
cv2.waitKey(0)

