import RPi.GPIO as GPIO
from time import sleep
from escpos import *
import sys
import json

BUTTON_ADD = 15
BUTTON_CANCEL = 14
BUTTON_DONE = 18
BUTTON_TARE = 4

from RPLCD import *
from RPLCD.i2c import CharLCD

# Initiate LCD using PCF8574 Character Codex and I2C Address 0x27
lcd = CharLCD('PCF8574', 0x27)

# Set Cursor Position to Row 0 Col 4
lcd.cursor_pos= (0, 4)
lcd.write_string('Yusfidah  TA')

# Set Cursor Position to Row 2 Col 5
lcd.cursor_pos= (2, 5)
lcd.write_string('Loading...')

for c in range(5):
    # Set Cursor Position to Row 3 Col c (range 0 - 5)
    lcd.cursor_pos= (3, c)
    lcd.write_string('*')
    # Delay 200ms
    sleep(0.2)

# Import TensorFlow
import tensorflow as tf

for c in range(5, 8):
    # Set Cursor Position to Row 3 Col c (range 5 - 8)
    lcd.cursor_pos= (3, c)
    lcd.write_string('*')
    # Delay 200ms
    sleep(0.2)

# Import OpenCV Python
import cv2 as cv
for c in range(8, 15):
    # Set Cursor Position to Row 3 Col c (range 8 - 15)
    lcd.cursor_pos= (3, c)
    lcd.write_string('*')
    # Delay 200ms
    sleep(0.2)

EMULATE_HX711=False

# Set Reference
'''
The correct calibration of the weight sensor and the Raspberry Pi balance is crucial.
For this we need a comparison object whose weight we know.
For example, I have taken two packs of rice (1kg each), since it is recommended to choose an average value of the maximum. Place it on the scale and run it again with `sudo python example.py`. The displayed values can be positive as well as negative. In my case were displayed at 2kg ( = 2000 gram) values around -1390000. My reference value is thus -1390000 ÷ 2000 = -695
'''
referenceUnit = -695

prices_file = open("prices.json")
prices = json.load(prices_file)
def getPrices(name):
    '''
    This function get data from JSON based on Label that be get from the 'name'
    '''
    return [
        dictionary for dictionary in prices
        if dictionary['name'] == name
    ][0]['prices']
    

if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711
else:
    from emulated_hx711 import HX711

# Setting HX711
# GPIO 5 = DATA
# GPIO 6 = SCK
hx = HX711(5, 6)
hx.set_reading_format("MSB", "MSB")
hx.set_reference_unit(referenceUnit)
hx.reset()
hx.tare()

# Setting class name of the list of inferences
# DO NOT CHANGE THE LISTS
class_names = ["Apel Fuji", "Apel Hijau", "Apel Merah", "Jeruk", "Mangga", "Pisang", "Strawberry"]

# Load Model from '/model_fruit_detector_v1'
loaded_model = tf.keras.models.load_model("model_fruit_detector_v1")

# Video capture from camera
cam = cv.VideoCapture(0)

# Defining GPIO Mode
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_ADD, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(BUTTON_CANCEL, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(BUTTON_DONE, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(BUTTON_TARE, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

for c in range(15, 20):
    # Set Cursor Position to Row 3 Col c (range 15 - 20)
    lcd.cursor_pos= (3, c)
    lcd.write_string('*')
    # Delay 200ms
    sleep(0.2)

# Clear the LCD
lcd.clear()

# Initiate Global Variable for Realtime Weighing
weight = 0

# Initiate Global List Variable for Cummulative Weighing and Detection
cummulative = []

if not cam.isOpened():
    print("error opening camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    # if frame is read correctly ret is True
    if not ret:
        print("error in retrieving frame")
        break
    # lcd.clear()
    
    lcd.cursor_pos= (0, 0)
    lcd.write_string('Terakhir: ')
    if len(cummulative) == 0:
        '''
        If Length of cummulative detection is 0
        The display will show empty history
        '''
        lcd.cursor_pos= (0, 10)
        lcd.write_string("          ")
        lcd.cursor_pos= (1, 10)
        lcd.write_string("          ")
    else:
        '''
        If Length of cummulative detection is greater than 0
        The display will show the history
        '''
        lcd.cursor_pos= (0, 10)
        lcd.write_string(cummulative[len(cummulative) - 1][0])
        lcd.cursor_pos= (1, 10)
        lcd.write_string(f"{cummulative[len(cummulative) - 1][1]}")
        lcd.cursor_pos= (1, 18)
        lcd.write_string("kg")
    
    weight = hx.get_weight(1)
    lcd.cursor_pos= (3, 0)
    lcd.write_string('Berat: ')
    lcd.cursor_pos= (3, 7)
    lcd.write_string(f"{abs(round(weight/1000, 1))} kg")
    lcd.cursor_pos= (3, 13)
    lcd.write_string('     ')
    
    if(GPIO.input(BUTTON_TARE) == GPIO.HIGH):
        '''
        Make the loadcell 0 again
        '''
        hx.set_reference_unit(referenceUnit)
        hx.reset()
        hx.tare()
    
    if(GPIO.input(BUTTON_CANCEL) == GPIO.HIGH):
        '''
        Cancel the previous detection and calculation
        '''
        if (len(cummulative) > 0):
            cummulative.pop()
        
    if(GPIO.input(BUTTON_DONE) == GPIO.HIGH):
        '''
        Print the cummulative detection and calculation
        '''
        p = printer.Usb(0x0fe6,0x811e , timeout=0, in_ep=0x81, out_ep=1)
        p.set()
        total_prices = 0;
        
        for single in cummulative:
            total_prices = total_prices + single[2]
            p.text(f"\n{pred_class}\t")
            p.text(f"{single[1]} Kg \n")
            p.text(f"Rp {single[2]}\n\n")
        
        p.text(f"Total Harga: \nRp {total_prices},-\n")
        p.text("-------------------------------\n")
        p.cut()
        p.close()
        cummulative = []
        
    
    if(GPIO.input(BUTTON_ADD) == GPIO.HIGH):
        '''
        Detect the name of fruit and get the weight and calculate the prices
        '''
        lcd.cursor_pos= (0, 10)
        lcd.write_string("          ")
        lcd.cursor_pos= (1, 10)
        lcd.write_string("          ")
        
        # Convert Image from BGR to RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Convert Image from Image to Tensors
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        # Resize Image from variative image to 224 x 224
        image = tf.image.resize(image, size=[224, 224])
        
        # Predict Tensors using loaded model
        pred = loaded_model.predict(tf.expand_dims(image, axis=0))
        
        if len(pred[0]) > 1:
            pred_class = class_names[pred.argmax()]
        else:
            pred_class = class_names[int(tf.round(pred)[0][0])]
        
        # Appending prediction and calculation data to cummulative
        cummulative.append([
            pred_class,
            round(weight/1000, 1),
            round(weight/1000, 1) * getPrices(pred_class)
            
        ])
        
        hx.power_down()
        hx.power_up()
        # time.sleep(0.1)
        
        # cv.putText(frame, "Prediksi: " + pred_class, (20, 40 -2),
                   #cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    # cv.imshow("Detection", frame)

   
    if cv.waitKey(1) == ord('q'):
      break

cam.release()
cv.destroyAllWindows()

