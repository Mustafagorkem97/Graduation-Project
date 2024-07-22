#define echoPin 13 //Ultrasonik sensörün echo pini Arduino'nun 13.pinine tanımlandı.
#define trigPin 12 //Ultrasonik sensörün trig pini Arduino'nun 12.pinine tanımlandı.
#define MotorR1 4 //L298N IN1 pini tanımlandı.
#define MotorR2 5 //L298N IN2 pini tanımlandı.
#define MotorRE 3 //L298N ENA pini tanımlandı.
#define MotorL1 6 //L298N IN3 pini tanımlandı.
#define MotorL2 7 //L298N IN4 pini tanımlandı.
#define MotorLE 9 //L298N ENB pini tanımlandı.
long sure, uzaklik; //süre ve uzaklık diye iki değişken tanımlıyoruz.

void setup() {
  // ultrasonik sensör Trig pininden ses dalgaları gönderdiği için OUTPUT (Çıkış),
  // bu dalgaları Echo pini ile geri aldığı için INPUT (Giriş) olarak tanımlanır.
  pinMode(echoPin, INPUT);
  pinMode(trigPin, OUTPUT); 
  pinMode(MotorL1, OUTPUT); //Motorlarımızı çıkış olarak tanımlıyoruz.
  pinMode(MotorL2, OUTPUT); //Motorlarımızı çıkış olarak tanımlıyoruz.
  pinMode(MotorLE, OUTPUT); //Motorlarımızı çıkış olarak tanımlıyoruz.
  pinMode(MotorR1, OUTPUT); //Motorlarımızı çıkış olarak tanımlıyoruz.
  pinMode(MotorR2, OUTPUT); //Motorlarımızı çıkış olarak tanımlıyoruz.
  pinMode(MotorRE, OUTPUT); //Motorlarımızı çıkış olarak tanımlıyoruz.
  Serial.begin(9600);

}

void loop() {
  digitalWrite(trigPin, LOW); //sensör pasif hale getirildi
  delayMicroseconds(5);
  digitalWrite(trigPin, HIGH); //Sensore ses dalgasının üretmesi için emir verildi
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW); //Yeni dalgaların üretilmemesi için trig pini LOW konumuna getirildi
  sure = pulseIn(echoPin, HIGH); //ses dalgasının geri dönmesi için geçen sure ölçülüyor
  uzaklik = sure / 29.1 / 2; //ölçülen süre uzaklığa çevriliyor
  Serial.println(uzaklik);
  if (uzaklik < 15) // Uzaklık 15'den küçük ise,
  {
    geri();  // 150 ms geri git
    delay(150);
    sag();  // 250 ms sağa dön
    delay(250);
  }
  else {  // değil ise,
    ileri(); // ileri git
  }

}

void ileri(){  // Robotun ileri yönde hareketi için fonksiyon tanımlıyoruz.
  digitalWrite(MotorR1, HIGH); // Sağ motorun ileri hareketi aktif
  digitalWrite(MotorR2, LOW); // Sağ motorun geri hareketi pasif
  analogWrite(MotorRE, 150); // Sağ motorun hızı 150
  digitalWrite(MotorL1, HIGH); // Sol motorun ileri hareketi aktif
  digitalWrite(MotorL2, LOW); // Sol motorun geri hareketi pasif
  analogWrite(MotorLE, 150); // Sol motorun hızı 150
  
  
}
void sag(){ // Robotun sağa dönme hareketi için fonksiyon tanımlıyoruz.
  digitalWrite(MotorR1, HIGH); // Sağ motorun ileri hareketi aktif
  digitalWrite(MotorR2, LOW); // Sağ motorun geri hareketi pasif
  analogWrite(MotorRE, 0); // Sağ motorun hızı 0 (Motor duruyor)
  digitalWrite(MotorL1, HIGH); // Sol motorun ileri hareketi aktif
  digitalWrite(MotorL2, LOW); // Sol motorun geri hareketi pasif
  analogWrite(MotorLE, 150); // Sol motorun hızı 150
  
  
}
void geri(){ // Robotun geri yönde hareketi için fonksiyon tanımlıyoruz.
  digitalWrite(MotorR1, LOW); // Sağ motorun ileri hareketi pasif
  digitalWrite(MotorR2, HIGH); // Sağ motorun geri hareketi aktif
  analogWrite(MotorRE, 150); // Sağ motorun hızı 150
  digitalWrite(MotorL1, LOW); // Sol motorun ileri hareketi pasif
  digitalWrite(MotorL2, HIGH); // Sol motorun geri hareketi aktif
  analogWrite(MotorLE, 150); // Sol motorun hızı 150
  
}
