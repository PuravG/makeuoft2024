#include <L298N.h>
#include <Rotary.h>

// Pin definition
#define encA 8
#define encB 9

#define EN 3
#define IN1 5
#define IN2 6

// With Enable pin to control speed
L298N motor(EN, IN1, IN2);

// Encoder
int counter = 0; 
Rotary rotary = Rotary(encA, encB);

// "PID"
int target = 30;
int minPower = 150;
int error = 100;
const double kP = 25.0;
int output;

bool loading = true; // state machine var

void handleEnc(){
  unsigned char result = rotary.process();
  if (result == DIR_CW) {
    counter++;
    // Serial.println(getPos());
    // Serial.print("error: ");
    // Serial.println(target - getPos());
    computePID();

  }
  else if (result == DIR_CCW) {
    counter--;
    computePID();
  }
}

int getPos(){
  return -counter;
}

void computePID(){
  error = target - getPos();
  output = kP*error;

  if(loading){
    handleMovement();
  }
  else{
    if(Serial.available()){ // Slap
      Serial.read();
      Serial.println("Slapped!");
      motor.forward();
      motor.setSpeed(255);
      delay(200);
      motor.stop();
    }
  }
  // Data logging
  Serial.println(getPos());
  Serial.print("error: ");
  Serial.print(error);
  Serial.print(" output: ");
  Serial.println(output);

}

void handleMovement(){
  
  if(error < 2){
    motor.stop();
    
    Serial.println("Target Reached!");
    loading = false;
  }
  else {
    if(output < minPower) output = minPower;
    // motor.forward();
    motor.setSpeed(output);
    p();
  }
}

void setup() {
  motor.forward();
  // motor.setSpeed(200);
  
  Serial.begin(9600);

}

int printTime = millis();

void p(){
  if (millis() - printTime > 100){
    Serial.print("ACC OUTPUT: ");
    Serial.println(output);
    printTime = millis();
  }
}

void loop() {
  // motor.stop();

  // if(Serial.available()){
  //   Serial.read();
  //   motor.forward();
  //   motor.setSpeed(255);
  //   delay(200);
  //   motor.stop();

  // }
  // delay(100);

  handleEnc();
  // handlePID();
  // handleMovement();

}
