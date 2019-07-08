# import the necessary packages
import cv2


def show_image(title, picture):
    cv2.imshow(title, picture)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def resize_image(picture):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = 200.0 / picture.shape[1]
    dim = (200, int(picture.shape[0] * r))
    # perform the actual resizing of the image and show it
    resize = cv2.resize(picture, dim, interpolation=cv2.INTER_AREA)
    show_image("resize", resize)


def rotate_image(picture):
    # grab the dimensions of the image and calculate the center
    # of the image
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    # rotate the image by 180 degrees
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(picture, M, (w, h))
    show_image("rotated", rotated)


def crop_image(picture):
    # crop the image using array slices -- it's a Numpy array
    # after all
    cropped = picture[70:170, 440:540]
    show_image("cropped", cropped)
    return cropped


# load the image and show it
image = cv2.imread("images/jurassic-park-tour-jeep.jpg")
show_image("original", image)

resize_image(image)
rotate_image(image)
cropped = crop_image(image)
cv2.imwrite("images/jurassic-park-tour-jeep.thumbnail.png", cropped)
