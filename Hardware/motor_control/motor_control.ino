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
int target = 60;
int minPower = 250;
int error = 100;
const double kP = 25.0;
int output;

int getPos(){
  return -counter;
}

void logData(){
  Serial.println(getPos());
  Serial.print("error: ");
  Serial.print(error);
  Serial.print(" output: ");
  Serial.println(output);
}

void handleEnc(){
  unsigned char result = rotary.process();
  if (result == DIR_CW) {
    counter++;
    logData();
  }
  else if (result == DIR_CCW) {
    counter--;
    logData();
  }

}

enum class StateTypes{
  loading,
  ready,  // loaded
  slapping
};

StateTypes state = StateTypes::loading;

void handlePID(){
  error = target - getPos();
  output = kP*error;

  switch(state){
    case StateTypes::loading:
      handleMovement();
      break;
    case StateTypes::ready:
      if(Serial.available()){ // Slap
        Serial.read();
        Serial.println("Slapped!");
        motor.forward();
        motor.setSpeed(255);
        delay(200);
        motor.stop();
        state = StateTypes::slapping;
      }
      break;
    case StateTypes::slapping:
      if(error > 30){
        motor.forward();
        state = StateTypes::loading;
      }
      break;    
  }
  // if(loading){
  //   handleMovement();
  // }
  // else{
  //   if(Serial.available()){ // Slap
  //     Serial.read();
  //     Serial.println("Slapped!");
  //     motor.forward();
  //     motor.setSpeed(255);
  //     delay(200);
  //     motor.stop();
  //     motor.forward();
  //     loading = true;
  //   }
  // }
  // Data logging


}

void handleMovement(){
  
  if(error < 2){
    motor.stop();
    
    Serial.println("Target Reached!");
    state = StateTypes::ready;
  }
  else {
    if(output < minPower) output = minPower;
    // motor.forward();
    motor.setSpeed(output);
    // p();
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
  handlePID();
  // handleMovement();

}
