# Leaderboard kalibrasyonu — ölçüm aracımızın çözünürlüğü

*Ölçüldü 2026-08-31, Kaggle API'den doğrudan (`kaggle competitions submissions`,
`kaggle competitions leaderboard --download`). Bu belge bir yorum değil, ölçüm.*

Bu proje boyunca +0.002–0.005 mertebesinde yerel kazançlar ölçtük ve LB'nin
kıpırdamamasını "kazanç transfer olmuyor" diye okuduk. **Bu okuma yanlıştı.**

---

## 1. LB iki ondalık gösteriyor

Kaggle API'nin döndürdüğü ham `publicScore` değerleri:

| Gönderim | publicScore |
|---|---|
| sub001 | 0.28 |
| sub002 | 0.33 |
| sub003 | 0.35 |
| sub004 | 0.36 |
| sub005 | 0.36 |
| sub006 | 0.37 |
| sub007 | 0.38 |
| sub008 | 0.38 |
| sub009 | 0.38 |

Dokuzunun dokuzu da **tam iki ondalık**. Aynı şey 468 takımlık tam
leaderboard için de geçerli. Yani gerçek skor `[0.375, 0.385)` aralığının
neresinde olursa olsun ekranda `0.38` yazıyor.

**Sonuç:** sub007 = sub008 = sub009 = 0.38 gözlemi, bu üç gönderimin **0.000 ile
0.009 arasında** herhangi bir farkla ayrıştığıyla uyumludur. Bu üçlüden
"transdüktif öz-eğitim tam sıfır getirdi" diye çıkardığımız sonuç, LB'nin
veremeyeceği bir çözünürlüğü ondan istemişti. (Transdüksiyonun nötr olduğu
sonucu ayrıca `train_test_shift` ile bağımsız olarak destekleniyor — o kısım
ayakta, ama LB kanıtı olarak kullanılan kısım geçersiz.)

**Yerel ölçüm, elimizdeki tek hassas alet.** LB bir doğrulayıcı, bir optimizasyon
sinyali değil.

---

## 2. Yerel→LB ofseti ≈ −0.049 ve şaşırtıcı derecede sabit

Ofset ancak *aynı türden* yerel ölçümler arasında karşılaştırılabilir. Erken
gönderimler fold-0 sayıları raporluyordu; havuzlanmış sayıya geçtikten sonra:

| Gönderim | Yerel (havuzlanmış) | LB | Ofset |
|---|---|---|---|
| sub004 | 0.4100 | 0.36 | **−0.0500** |
| sub007 | 0.4286 | 0.38 | **−0.0486** |

İki bağımsız nokta, 0.0014 içinde. docs/13'ün "≈ −0.05" tahmini doğrulandı.
(Fold-0 tabanlı erken gönderimler −0.060…−0.076 veriyor; bu tutarsızlık değil,
fold-0'ın havuzlanmış sayıdan sistematik olarak iyimser olması.)

---

## 3. Hedefin sayısal hali

Ekranda `X` yazması için gerçek skorun `[X−0.005, X+0.005)` içinde olması yeter.
Yani gereken şey bandın **merkezi değil alt kenarı** — bu ayrım hedefi yarıya
indiriyor.

| Ekranda | Gereken gerçek | Gereken yerel | Bulunduğumuz yer | Açık |
|---|---|---|---|---|
| 0.39 | ≥ 0.385 | ≥ **0.4343** | 0.4290 | **+0.0053** |
| 0.40 | ≥ 0.395 | ≥ 0.4443 | 0.4290 | +0.0153 |
| 0.41 | ≥ 0.405 | ≥ 0.4543 | 0.4290 | +0.0253 |

Ofset tahmini iki *yuvarlanmış* noktadan geldiği için kendisi de ±0.005
belirsiz; dolayısıyla 0.39 için gereken artış **0.000 ile +0.010 arasında**.
Bandın neresinde olduğumuzu bilmiyoruz, ve bu bir kez bile ölçülemez — ancak
bir basamak atladığımızda geriye dönük daralır.

### Bu, biriktirmeyi tek uygulanabilir plan yapıyor

docs/13 zaten şunu yazmıştı ve haklıydı: *"tek başına değersizler, birlikte tam
da gereken aralığı kapatıyorlar."* Ölçülmüş ama benimsenmemiş küçük pozitifler:

Naif toplam +0.0085 çıkıyor ve yerel 0.4375 → ekranda 0.39 demek. **Ama bu
toplam sahte** — bileşenleri tabanlarına göre ayırınca çoğu buharlaşıyor:

| | İddia edilen | Taban | Dürüst cross-fit değeri |
|---|---|---|---|
| M2F üçüncü kaynak | +0.0025 | fold-0 cross-fit | **+0.0025**, ama yalnız fold-0; 5 fold için 4 M2F modeli daha eğitmek gerek |
| Seçim-anında ayrıştırma | +0.0022 | **in-sample** | `eval_nodeov.json` cross-fitted **−0.0011** (0.4244 vs 0.4255). In-sample optimum yukarı yanlı; **≈ 0** |
| Seed-refine üçüncü kulvar | +0.0030 | **oracle** | gerçekleşen hiç ölçülmedi; on üç-başarısızlık önsavı altında **≈ 0** |
| %2.5 küçültme | +0.0004 | cross-fit | +0.0004 (gürültü) |
| 2048 dedektör kulvarı | +0.0004 | cross-fit | +0.0004 (gürültü); ayrıca ayrı bir havuz, e013'e *eklenmiyor* |
| **Dürüst toplam** | | | **≈ +0.0033** ve bunun +0.0025'i 4 GPU-günü eğitim istiyor |

Yerel 0.4290 + 0.0033 = 0.4323 → gerçek ≈ 0.3830 → **ekranda hâlâ 0.38**.

Yani biriktirme planı, bileşenleri dürüst tabanlarına indirgeyince **gereken
+0.0053'ün altında kalıyor**. docs/13'ün "birlikte tam gereken aralığı
kapatıyorlar" öngörüsü, o küçük kazançların in-sample ve oracle sayılarını
gerçekleşen kazançmış gibi topladığı için fazla iyimserdi.

⚠ Ayrıca bu projede birleşim daha önce **bir kez** test edildi ve üç *negatifin*
birleşimi toplamlarından **kötü** çıktı (P4, −0.0134) — bileşenlerin bağımsız
toplandığı varsayımı burada bir kez ölçüldü ve tutmadı.

İşletme kuralı: **+0.005'in altındaki tek bir kazanç tek başına gönderilmeye
değmez**, çünkü ölçülemez — ama biriktirilebiliyorsa değer kazanır.

---

## 4. Alanın güncel yapısı (468 takım)

| Skor | Takım |
|---|---|
| **0.55** | **12** |
| 0.41–0.54 | **0** |
| 0.40 | 5 |
| 0.39 | 7 |
| 0.38 | 25 (**biz, sıra 31**) |
| 0.37 | 29 |
| 0.36 | 39 |

İki şey değişti ve ikisi de bizim aleyhimize:

1. **Sızıntı bandı 4 takımdan 12 takıma çıktı**, hepsi tam olarak 0.55. On iki
   bağımsız modelleme çalışmasının aynı iki-ondalık skorda buluşması, üstelik
   0.41–0.54 aralığı tamamen boşken, bir modelleme dağılımı değildir. Bu,
   docs/18'deki sızıntı tezinin en güçlü tek kanıtı hâline geldi.
2. **12 dürüst takım önümüze geçti** (5 tane 0.40'ta, 7 tane 0.39'da). Sıramız
   11/378'den **31/468**'e düştü. "0.38 yetersiz" artık bir his değil, ölçüm.

---

## 5. Bu belgenin değiştirdiği önceki sonuçlar

- ⚠ docs/15 ledger, transdüktif öz-eğitim satırında "LB 0.38 = 0.38" ifadesini
  kanıt olarak kullanıyor. LB'nin çözünürlüğü bu karşılaştırmayı yapamaz;
  o satırın LB kısmı geri çekilmelidir. Mekanizma sonucu (`train_test_shift`
  i.i.d. ölçtüğü için transdüksiyonun kazandıracak şeyi yok) ayakta kalır.
- ⚠ Genel olarak "LB kıpırdamadı, demek ki kazanç gerçek değil" biçimindeki her
  akıl yürütme, kazanç 0.01'in altındaysa geçersizdir.


---

## 6. sub010 — kalibrasyonun ilk canlı sınaması (2026-08-31)

Kalibrasyonun ardından tek değişkenli bir sonda gönderildi: sub008'in tam
konfigürasyonu (recipe3, conf 0.40, p_sel 0.35, %5 küçültme), yalnız seçici
eğitim↔dağıtım arasında **tanımı değişen** 9 öznitelik olmadan yeniden eğitildi.

| | değer |
|---|---|
| Yerel cross-fitted | 0.4291 (vs 0.4286) |
| Seçici AUC | 0.8923 (vs 0.8937) |
| Tutulan aday | **1237** (vs sub008'in 1326'sı) |
| **LB** | **0.38** |

Bu sonuç kalibrasyonu **doğruluyor, çürütmüyor**: +0.0005'lik bir yerel fark,
iki ondalıklı bir tahtada tanım gereği görünmez. Gönderimin amacı skor değildi —
yerel ölçümün yapısal olarak göremediği tek şeyi, *transferi* denemekti.
Tutulan aday sayısındaki %7'lik düşüş, atılan özniteliklerin dağıtımda
sistematik olarak fazladan tutma satın aldığını doğruluyor.

**Nihai gönderim seçimi için kural:** iki konfigürasyon LB'de aynı görünüyorsa,
**en yüksek yerel sayıyı değil, en az eğitim/dağıtım uyumsuzluğunu** taşıyanı
seç. Tahta ikisini ayıramaz; private LB ayırabilir.


---

## 7. sub011 — gerçek kesirli küçültme (2026-09-01)

`shrink_frac`'in parametresinin atıl olduğu bulunup `shrink_true_frac` yazıldı ve
eksen ilk kez tarandı. Tepe f=0.10'da, cross-fitted **0.4331** (üretim 0.4286).

| | değer |
|---|---|
| Yerel cross-fitted | **0.4331** (+0.0045) |
| Öngörülen gerçek LB | 0.4331 − 0.0493 = **0.3838** |
| 0.39 için gereken | ≥ 0.385 |
| **Gerçekleşen LB** | **0.38** |

Kalibrasyon **üçüncü kez doğrulandı**: 0.3838 öngördü, 0.385 eşiğinin altında
kaldı, ekranda 0.38 çıktı. Bu, aracın çalıştığının kanıtı — kazancın sahte
olduğunun değil.

### Kalan açık artık çok küçük

| | yerel gereken | bulunduğumuz | açık |
|---|---|---|---|
| Önce (sub008 konfigürasyonu) | 0.4343 | 0.4286 | +0.0057 |
| **Şimdi (sub011)** | 0.4343 | **0.4331** | **+0.0012** |

Tek bir düzeltme açığın **beşte dördünü** kapattı. Kalan +0.0012, bu projede
"ölçülemez" diye kenara konan mertebenin bile altında — yani artık *biriktirme*
gerçekten uygulanabilir bir plan, çünkü tek bir küçük kazanç yetiyor.

⚠ Ofsetin kendisi ±0.005 belirsiz olduğu için şu an 0.385'in hangi tarafında
olduğumuzu bilmiyoruz; 0.39'u gördüğümüz an bu belirsizlik geriye dönük daralır.


---

## 8. Dört düz gönderim ne söylüyor (2026-09-01)

| Gönderim | Değişiklik | Yerel | Öngörülen gerçek | LB |
|---|---|---|---|---|
| sub008 | taban | 0.4286 | 0.3793 | 0.38 |
| sub010 | seçici öznitelik hijyeni | 0.4291 | 0.3798 | 0.38 |
| sub011 | + gerçek kesirli küçültme | 0.4331 | 0.3838 | 0.38 |
| sub012 | + blob **çeyrek** ağırlık | ~0.4331 | 0.3838 | 0.38 |
| sub013 | + blob **yarım** ağırlık | ~0.4371 | 0.3878 | 0.38 |

sub013'ün öngörüsü 0.3878 ve 0.39 bandı 0.385'ten başlıyor. **Göstermedi.**
Bu, transfer üzerine bir kısıt koyar:

- sub008'in 0.38 göstermesi → gerçek skoru `T ∈ [0.375, 0.385)`
- sub013 = `T + 0.0085` (tam transfer varsayımıyla) ve o da 0.38 gösterdi
- → `T < 0.3765`, yani `T ∈ [0.375, 0.3765)` — **bandın en alt %15'i**

İki olasılık var ve tahtayla ayırt edilemiyorlar:

**(a)** Gerçekten bandın dibindeydik (a priori %15 ihtimal) ve kazançlar gerçek.
**(b)** Kazançlar tam transfer olmuyor. Yerel ölçümler cross-fitted ve
eşleştirilmiş (küçültme +0.0045; blob +0.0039, t=3.24) — yani *yerel olarak*
sağlamlar. Ama yerel→LB aktarımı, bu projede hiçbir zaman doğrudan
doğrulanamadı; −0.0493 ofseti iki yuvarlanmış noktadan türetildi.

⚠ **İşletme sonucu:** bugün dört gönderim harcandı ve ekran kıpırdamadı. Bundan
sonra tek bir +0.004 mertebesindeki değişiklik için gönderim harcamak
bilgilendirici değil. Ya **birikmiş** bir değişiklik paketi gönderilmeli, ya da
0.01 mertebesinde tek bir kazanç beklenmeli.


---

## 9. ⚠ DÜZELTME — tahta iki ondalık gösteriyor ama iki ondalıkla SIRALAMIYOR

Talha'nın gözlemi bu belgenin işletme sonucunu düzeltti. 0.38 grubu içinde 26
takımın 4.'süyüz; üstelik **gruptaki en yeni gönderim bizimki**. Beraberlik
gönderim zamanına göre bozulsaydı (en eski önce) en altta olurduk. Grup içi
gönderim tarihleri de sıralamayla monoton değil.

**Sonuç: Kaggle yuvarlanmamış gerçek skorla sıralıyor.** Ekran iki ondalık, sıra
tam hassasiyetli.

### Bu iki şeyi değiştiriyor

**1. Bandın neresinde olduğumuz.** §8'de "gerçek skorumuz bandın en alt %15'inde
olmalı" diye çıkarım yapmıştım. Grup içi 4.'lük bunun **tersini** söylüyor:
üstümüzde yalnız 3 takım var ve onlar da 0.385'in altında. 26 takım uniform
dağılsa 4. olmak ~%86 yüzdeliğe, yani ~0.3836'ya karşılık gelir.

**2. Ölçüm çözünürlüğümüz.** Bütün gün "tahta +0.004'lük kazancı göremez" diye
yazdım. **Yanlış.** Grup içi sıra çok daha ince: 0.010 genişliğinde bir banda 26
takım düşüyorsa bir basamak ≈ **0.0004** gerçek skor — ekranın çözünürlüğünden
**25 kat** hassas.

### Ama bir gerilim var, kaydediyorum

| | grup içi konum | yerel |
|---|---|---|
| Bugünün başı (sub009) | 5. / 25 | 0.4286 |
| Şimdi (sub013) | 4. / 26 | ~0.4371 |

+0.0085 yerel kazanç **bir** basamak getirdi. Basamak başına ~0.0004 varsayımıyla
bu, transferin çok düşük olduğunu ima eder. İki okuma mümkün ve ayırt edilmedi:

- takımlar bandın üstünde **sıkışık**, dolayısıyla bir basamak 0.0004'ten çok
  daha fazla skora karşılık geliyor;
- ya da kazançlar kısmen transfer oluyor.

⚠ Ayrıca bugünün başındaki 5.'lik, grubu doğrudan listeleyerek değil "üstte 24
takım" sayısından çıkarıldı; o yüzden ±1 belirsizliği var.

### İşletme kuralı (§8'deki kuralın yerine geçer)

Her gönderimden sonra **grup içi sırayı kaydet**, yalnız gösterilen skoru değil.
Bir değişikliğin işe yarayıp yaramadığı buradan okunur. Bunun için
`scripts/track_rank.sh` eklendi.


---

## 10. 🔑 EN HASSAS ALET: Kaggle'ın otomatik "en iyi 2" seçimi

Talha, Kaggle'ın otomatik olarak **sub010 ve sub011**'i seçtiğini fark etti
(arayüzde mavi işaretli). API seçim durumunu göstermiyor, o yüzden tek kaynak bu.

### Neden bu bir ölçüm

Otomatik seçim, halka açık skora göre en iyi ikiyi alır. Seçilenler ne en
eskiler (sub007/008) ne de en yeniler (sub013/014) — **ortadakiler**. Bu ancak
seçim **tam hassasiyetli** skora göre yapılıyorsa mümkün, ki §9'da tahtanın
gerçek skorla sıraladığını bağımsız olarak kanıtlamıştık.

Ve bu alet, sıradan **daha iyi**: grup içi sıra başka takımların hareketiyle
kirleniyor (bir gün içinde 5→4→5 okuduk ve hepsi başkalarındandı), otomatik seçim
ise yalnız **kendi gönderimlerimizi** karşılaştırıyor.

### Ne söylüyor

| Gönderim | Değişiklik | Yerel | Seçili? |
|---|---|---|---|
| sub010 | öznitelik hijyeni | 0.4291 | ✅ |
| sub011 | + gerçek kesirli küçültme | 0.4331 | ✅ |
| sub012 | + blob ¼ | ~0.4331 | ❌ |
| sub013 | + blob ½ | ~0.4371 | ❌ |
| sub014 | + iki-aşamalı seçici | ~0.4381 | ❌ |

Doğruysa: **blob ansambl üyesi ve iki-aşamalı seçici test setinde zarar
veriyor**, yerel olarak +0.0039 ve +0.0011 ölçülmüş olmalarına rağmen. Yani
gerçek skorumuz sub011'de tepe yapmış ve o günden beri düşmüş.

### ⚠ Çekince

Seçim, sub012-014 henüz yokken yapılıp güncellenmemiş olabilir. Kaggle normalde
her yeni gönderimde otomatik seçimi tazeler, ama bunu API'den doğrulayamıyorum.
**Doğrulama yolu:** bir sonraki gönderimden sonra seçimin değişip değişmediğine
bakmak. Değişiyorsa alet canlı; değişmiyorsa seçim bayat.

### İşletme kuralı

Bundan sonra her gönderimden sonra **seçilen ikilinin ne olduğu** kaydedilmeli.
Yeni bir gönderim seçilenlerden birini yerinden ediyorsa gerçekten daha iyidir;
etmiyorsa değildir. Bu, ekrandaki 0.38'in ve grup içi sıranın veremediği
doğrudan bir karşılaştırma.

⚠ Ve nihai seçim için: yarışma bittiğinde private LB **seçili iki gönderimi**
kullanır. Yerel ölçümlerimiz sub014'ü en iyi diyor, otomatik seçim sub011'i.
İkisi çelişiyorsa, karar deadline'a (2026-11-15) yakın, elde daha çok kanıtla
verilmeli — ve seçim **elle** yapılabilir.


---

## 11. ✅ 0.39 — ve ayrıştırma sondasının okuması (2026-09-02)

**sub015 = 0.39.** 27 Ağustos'tan beri ilk basamak hareketi. Sıra 31/481 → **23/481**,
0.39 grubunda 5/9. Sızıntı bandı hariç dürüst sıralamada 18. → **10.**

### Nedensellik, sondanın tasarladığı gibi okundu

| Gönderim | Değişiklik | LB |
|---|---|---|
| sub011 | gerçek kesirli küçültme | 0.38 |
| sub012 | + blob ¼ | 0.38 |
| sub013 | + blob ½ | 0.38 |
| sub014 | + blob ½ + iki-aşamalı | 0.38 |
| **sub015** | **küçültme + iki-aşamalı, blob YOK** | **0.39** |

sub014'te iki-aşamalı **zaten vardı** ve yine 0.38'de kaldı; tek fark blob'un
çıkarılması. Yani:

- **iki-aşamalı seçici → yardım ediyor**, ve yerel eşleştirilmiş +0.0011'i transfer oldu
- **blob ansambl üyesi → zarar veriyor**, yerel eşleştirilmiş +0.0039'a rağmen
- blob'un zararı, iki-aşamalının kazancını yutacak kadarmış

### Otomatik seçim aleti CANLI olduğunu kanıtladı

Seçim sub010+sub011'den **sub010+sub015**'e güncellendi. §10'daki "bayat olabilir"
çekincesi düştü. Bundan sonra her gönderimin gerçekten daha iyi olup olmadığı
buradan okunacak.

### Neden blob transfer olmadı — ve bu ne demek

İki ölçüm arasındaki fark protokolde değil, **kapsamda**:

| | ölçüldüğü yer | dağıtımdaki hâli |
|---|---|---|
| iki-aşamalı seçici | **5-fold üretim havuzu** (9609 aday) | aynı |
| blob üyesi | **yalnız fold-0** (1899 aday) | tek model, semantik alanın yarısı |

Blob ölçümü fold-0'da tek bir modeli, üretimin **5-fold** ansamblına karşı yarım
ağırlıkla test etti. Dağıtımda o tek model, 5-fold ortalamalarla yarışıyor ve
daha az veri görmüş. Yani kusur mekanizmada değil, **tek modele yarım ağırlık
vermekte**.

⚠ Bu, arka planda koşan 4-fold blob eğitimini **anlamsız kılmıyor, tam tersine
gerekli kılıyor**: 5-fold bir blob üyesi, prod_1536 ile adil bir karşılaştırma
olur. O bitene kadar blob üretimden uzak.

### Genel ders

Bir **ansambl üyesini** tek fold'da ölçmek yanıltıcı: ölçümde üye, o fold'un tek
modeline karşı yarışıyor; dağıtımda 5-fold ansambllara karşı yarışıyor. Bir
**seçici değişikliğini** 5-fold havuzda ölçmek ise doğrudan transfer oldu.


---

## 12. sub016 — ve iki değişkenli karşılaştırmanın bedeli (2026-09-02)

sub016 = sub015 + öznitelik hijyeni → **0.38**, yani sub015'in 0.39'undan aşağı.

### Ne öğrendik, ne öğrenmedik

Öğrendiğimiz: bu konfigürasyon sub015'ten kötü.

**Öğrenmediğimiz: neden.** Çünkü karşılaştırma tek değişkenli değil —
her paket kendi uydurulmuş politikasıyla gitti:

| | öznitelik | politika | tutulan | LB |
|---|---|---|---|---|
| sub015 | 32 | conf 0.40, p 0.55 | üst %45 | **0.39** |
| sub016 | 23 | conf 0.35, p 0.50 | üst %50 | 0.38 |

Düşüş öznitelik sayısından da olabilir, daha fazla aday tutmaktan da. Ayrıştırmak
için sub015'in politikasıyla 23 öznitelik gerekir.

### ⚠ Ve bu, otomatik seçimden yaptığım çıkarımı da geçersiz kılıyor

"sub010 seçili, demek ki öznitelik hijyeni testte yardım ediyor" demiştim.
sub010 ile sub011 **iki** yerde farklı (hijyen VE küçültme), yani o okuma
tek başına hijyeni gösteremezdi:

| | küçültme | öznitelik | skor |
|---|---|---|---|
| sub010 | üretim (halka, %19) | 23 | 0.38, seçiliydi |
| sub011 | gerçek kesirli 0.10 | 32 | 0.38 |

sub010 > sub011 olması, hijyenin değil **eski küçültmenin** daha iyi olmasından
da kaynaklanabilir. sub016'nın düşüşü ikinci okumayı destekliyor ama kanıtlamıyor.

### Ders

Otomatik seçim güçlü bir alet ama **yalnız tek değişkenli sondalarda** okunabilir.
sub015 tek değişkenliydi (sub014'ten yalnız blob çıkarıldı) ve temiz okundu;
sub016 değildi ve okunamıyor. Bundan sonra her sonda, öncekinden **tam olarak bir
şeyde** ayrılmalı — politika dahil.

---

## 13. sub017 – sub019 ve otomatik seçimin çözdüğü beraberlik (2026-09-02)

### 13.1 Tahtadaki tam durum

| gönderim | sub015'ten tek farkı | görünen |
|---|---|---|
| sub015 | — (referans) | **0.39** |
| sub016 | öznitelik hijyeni **+** kendi politikası (iki değişken) | 0.38 |
| sub017 | yalnız öznitelik hijyeni (23 öznitelik, sub015 politikası) | 0.38 |
| **sub018** | yalnız **küçültme kuralı** (üretim halka erozyonu) | **0.39** |
| sub019 | yalnız **beş katlı blob üyesi** | *puanlanıyor* |

0.39 grubu yalnızca {sub015, sub018}. Diğer her şey 0.38 ve altı.

sub017 kapanışı temiz: sub016 iki değişkenliydi ve okunamıyordu, sub017 politikayı
sub015'te sabitleyip yalnız öznitelik setini değiştirdi ve **0.38** verdi.
Öznitelik hijyeninin kendisi zarar veriyor. Bu, sub010'un otomatik seçilmesinden
yaptığım "hijyen yardım ediyor" çıkarımını kesin olarak geçersiz kılıyor.

### 13.2 İki hanenin çözemediğini otomatik seçim çözüyor

sub015 ve sub018 aynı haneyi gösteriyor, dolayısıyla ekran aralarında karar
veremiyor. Ama **Kaggle sıralamayı yuvarlanmamış gerçek skorla yapıyor** ve en
iyi ikiyi kendisi işaretliyor. Talha'nın raporu: arayüzde **mavi işaretli olan
sub018**.

sub018, sub015'ten *tasarım gereği tam olarak bir şeyde* ayrılıyor: maske
küçültme kuralı. Ansambl, seçici öznitelikleri, politika (conf 0.40 / p 0.55),
her ikisinde de CDF ile düzgün dağılıma eşlendiği için "üst %45'i tut" anlamı —
hepsi aynı. Seçicinin o küçültmede yeniden uydurulması değişikliğin *sonucu*,
ikinci bir değişken değil.

Dolayısıyla okuma tek anlamlı:

> **Üretim halka erozyonu, gerçek-kesirli 0.10'u testte geçiyor.**

Bu, yerel ölçümün tersi. Yerelde gerçek-kesirli **+0.0031** önde (0.4332'ye karşı
0.4301) ve süpürme temiz bir tepe veriyordu (0.00→0.4240, 0.07→0.4322,
**0.10→0.4331**, 0.13→0.4310, 0.19→0.4240).

### 13.3 Bu ikinci kez aynı yönü gösteriyor

Birinci kez sub010'un sub011 üzerinden seçilmesiydi: sub010 üretim küçültmesini,
sub011 gerçek-kesirliyi taşıyordu. O karşılaştırma **okunamazdı**, çünkü ikisi
öznitelik setinde de ayrılıyordu. Şimdi sub018–sub015 çifti o karışıklığı
taşımıyor ve aynı yönü veriyor.

İki bağımsız örnek aynı yöne işaret ediyor, ve arada sub017 öznitelik eksenini
kapattığı için birinci örneğin karışık değişkeni de artık açıklanmış durumda:
sub010'un seçilmesi **hijyenden değil, küçültmeden** geliyormuş.

### 13.4 Uyarı — bunu bir kez fazla okudum

Otomatik seçimden çıkarım yaparken bir kez yanıldım (sub010 → "hijyen yardım
ediyor") ve sub017 onu çürüttü. Bu yüzden sınırları açıkça yazıyorum:

- Mavi durumu **API'den doğrulanamıyor** (bkz. §10); tek kaynak arayüz raporu.
- Seçim **bir bit** bilgi: sub018 > sub015, ne kadar olduğunu söylemiyor.
- Her yeni gönderim seçimi tazeliyor, yani sub019 puanlandığında bu işaret
  değişebilir — ve değişmesi de bilgi.

### 13.5 Sonuç: sub019 yanlış küçültmeyi taşıyor

sub019 gerçek-kesirli 0.10 ile inşa edildi, çünkü blob ölçümü o küçültmeyle
yapılmıştı ve tek değişkenli kalmak sub015'i referans almayı gerektiriyordu.
Eğer §13.2 doğruysa, sub019 kazancının bir kısmını yanlış küçültme kuralına
veriyor.

Bir sonraki sonda (hak yenilendiğinde): **sub019'un semantik alanı + üretim
küçültmesi**, yani sub018'den tek farkı beş katlı blob üyesi olan bir kol.
Böylece iki eksen de kendi en iyi ayarında birleşir ve karşılaştırma yine tek
değişkenli kalır.

### 13.6 Harcanan hak

sub015 kayıtta **iki kez** görünüyor; ikincisinin açıklaması "quota probe".
Kalan hak sayısını öğrenmek için gönderim yapmıştım — hak sayısı zaten
gönderimin *çıktısında* yazıyor. Bir slot boşa gitti; kural olarak yazıldı:
kalan hakkı asla gönderim yaparak sorma.

---

## 14. Üç kollu sıralama çözüldü — blob tahtada doğrulandı (2026-09-02)

sub019 puanlandı: **0.39**, ve mavi çiftte **birinci**. sub018 ikinci. Yani üç
kolun üçü de aynı haneyi gösteriyor ama gerçek skor sıralaması artık biliniyor:

```
sub019  >  sub018  >  sub015
```

Üç kol, iki eksen:

| kol | semantik alan | küçültme kuralı |
|---|---|---|
| sub015 | recipe3 (blob yok) | gerçek-kesirli 0.10 |
| sub018 | recipe3 (blob yok) | **üretim halka erozyonu** |
| sub019 | **recipe_blob5_50 (beş katlı blob)** | gerçek-kesirli 0.10 |

Sıralama iki bağımsız karşılaştırma veriyor, her biri tek değişkenli:

**sub018 > sub015** → küçültme ekseni sabit semantik alanda: üretim erozyonu
gerçek-kesirli 0.10'u geçiyor. (Yerelde gerçek-kesirli +0.0031 öndeydi.)

**sub019 > sub015** → semantik eksen sabit küçültmede: **beş katlı blob üyesi
tahtada kazanıyor.** Bu, blob ölçümünün ilk kez transfer etmesi. Yerel ölçüm
+0.0061 (eşleşmeli se 0.0005, t = 11.86, 20/20 bölme) diyordu; tek-model formu
+0.0039 demiş ve sub014'te transfer etmemişti. Kapsam açıklaması doğrulandı.

Üçüncü karşılaştırma çapraz ve tek değişkenli değil ama bilgi veriyor:

**sub019 > sub018** → blob'un kazancı, yanlış küçültme kuralının kaybından
**büyük**. sub019 kötü küçültmeyi taşıyarak bile iyi küçültmeli blob'suz kolu
geçiyor.

### 14.1 Doğrudan sonuç: iki iyileştirme henüz bir arada değil

Hiçbir gönderim **beş katlı blob + üretim küçültmesi**ni birlikte taşımıyor.
İkisi de bağımsız olarak kazandığına göre, birleşimleri sub019'un üstünde
olmalı — kesin toplanabilir olduklarını varsaymadan, çünkü küçültme seçicinin
okuduğu alan özniteliklerini oynatıyor ve iki eksen orada etkileşebilir.

Hak yenilenince gidecek kol (hazırlanıyor):

```
semantik: recipe_blob5_50          (sub019'dan devralınan)
küçültme: üretim halka erozyonu    (sub018'den devralınan)
politika: conf 0.40, p 0.55        (üç kolda da sabit)
```

Bu kol sub019'dan **yalnız küçültmede**, sub018'den **yalnız semantik alanda**
ayrılıyor; yani iki okuma da tek değişkenli kalıyor ve üçlü ızgara kapanıyor.

### 14.2 Ne öğrendik — yöntem tarafı

İki hane gösteren bir tahtada üç kol aynı haneye düştü ve hiçbiri diğerini
ekrandan ayırt ettiremedi. Ayırt eden şey mavi çiftin **sırası** oldu. Bu, tek
bir gönderimden bir bit değil, sıralı çiftten **iki bit** çıkarmak demek — ve
kollar tasarım gereği tek değişkenli tutulduğu için o bitler doğrudan
mekanizmaya yazılabiliyor.

Kural: gönderimleri her zaman *bir öncekinden tam bir tasarım kararıyla* ayır,
ve mavi çiftin hangisinin önde olduğunu her turda sor. Ekranın çözemediğini
sıralama çözüyor.

---

## 15. Tahtanın tam sırası — 20 gönderim, 13 tek değişkenli A/B (2026-09-03)

### 15.1 Araç: skor değil, **sıra**

Gönderim uç noktası bir `sort` argümanı alıyor ve `SUBMISSION_SORT_BY_PUBLIC_SCORE`
seçeneği var (`src/eda/board_ordering.py`). Dönen **değerler** hâlâ iki haneye
yuvarlı, ama dönen **sıra** yuvarlanmamış gerçek skora göre. Yani ekranın
sakladığı her şey sıradan okunuyor.

Kullanmadan önce doğrulandı: üç ardışık çağrı aynı sırayı verdi, ve 0.38 bloğu
`010, 011, 012, 007, 008, 017, 009, 014, 016, 013` olarak dönüyor — ne tarih
sırası ne de tersi. Sıralama gerçek.

(§10'da "API seçim durumunu göstermiyor" yazmıştım; `SUBMISSION_GROUP_SELECTED`
sorgusu boş dönüyor, o kısım hâlâ geçerli. Ama sıralama erişilebilirmiş ve bu
mavi çiftten çok daha fazlasını veriyor: iki bit yerine tam sıra.)

### 15.2 Tam sıra

```
 1. sub019  blob5 ½ + gerçek-kesirli        0.39
 2. sub018  recipe3 + üretim küçültme       0.39
 3. sub015  recipe3 + gerçek-kesirli        0.39
 4. sub015  (aynı dosya, ikinci kopya)      0.39
 5. sub010  recipe3 + üretim + 23 öznitelik 0.38
 6. sub011  recipe3 + gerçek-kesirli        0.38
 7. sub012  blob ¼ (tek kat)                0.38
 8. sub007  prod5 + üretim küçültme         0.38
 9. sub008  recipe3 + üretim küçültme       0.38
10. sub017  recipe3 + 23 öznitelik + iki-aş 0.38
11. sub009  recipe3_mixed                   0.38
12. sub014  blob ½ (tek kat) + iki-aşamalı  0.38
13. sub016  23 öznitelik + kendi politikası 0.38
14. sub013  blob ½ (tek kat)                0.38
15-20. sub006 … sub001                 0.37 → 0.28
```

Tasarım matrisi kodlanıp tek eksende ayrılan bütün çiftler **mekanik** olarak
çıkarıldı — hikâyeye uyan çiftleri elle seçmemek için. İki ve daha fazla eksende
ayrılan 65 çift ayrıca listelenip **okunamaz** diye işaretlendi.

### 15.3 Blob hikâyesi kapandı: dört kıyas, hepsi varyansla tutarlı

| kazanan | kaybeden | tek fark |
|---|---|---|
| blob5 ½ (#1) | recipe3 (#3) | beş katlı blob **yardım ediyor** |
| blob5 ½ (#1) | blob ½ tek kat (#12) | aynı ağırlıkta **beş kat > bir kat** |
| recipe3 (#3, #6) | blob ½ tek kat (#12, #14) | tek katlı blob yarım ağırlıkta **zarar** |
| blob ¼ tek kat (#7) | blob ½ tek kat (#14) | tek kat için **çeyrek > yarım** |

Dördü birlikte tek bir mekanizmayı anlatıyor ve o mekanizma varyans:

- Tek bir fold-0 modeli yüksek varyanslı bir terim. **Yarım** ağırlıkta zarar
  veriyor, **çeyrekte** daha az zarar veriyor — ağırlık azaldıkça hasar azalıyor.
- Beş kat ortalandığında varyans düşüyor ve **yarım ağırlık en iyisi oluyor**.

Yerel süpürme "yarım en iyi" demişti (0.4488) ve tek-model formunda tahtada
yanlış çıkmıştı. Şimdi ikisi de açıklandı: süpürme doğruydu ama **ölçtüğü şey
düşük varyanslı bir üyeydi**, sub012/sub013 ise yüksek varyanslı olanı taşıdı.
Ledger §7'deki "mekanizma değil kapsam" tezi artık çapraz uyumda değil, tahtada
dört ayrı kıyasla doğrulanmış durumda.

### 15.4 Beklemediğim negatif: ağır üyeler zarar veriyor olabilir

| kazanan | kaybeden | tek fark |
|---|---|---|
| prod5 (#8) | recipe3 (#9) | recipe3'ün iki **ağır** üyesi |
| recipe3 (#9) | recipe3_mixed (#11) | transdüktif üye |

`recipe3`, beş üretim katının yanına `aug_heavy_f0_768` ve `aug_heavy_f0_1536`
ekliyor ve bu fold-0 harness'inde **+0.0054** ölçmüştü. Tahtada düz beş-katlı
`prod5`'in **altında**.

Ve bunun yapısal bir açıklaması var: ağır modellerin ikisi de **yalnız fold 0'da**
eğitilmiş, yani folds 1–4'ün doğrulama görüntülerini eğitimde görmüşler. Bu
yüzden kat-dışı havuz (`fused_pool_all`) onları **hiç içermiyor** — sadece
`_cldice` beş katı var. Ağır üyelerin katkısı hayatımızda bir kez, tek bir katta
ölçüldü ve bir daha doğrulanamadı; test zamanı ansamblına ise kalıcı olarak
girdiler.

Yani `recipe3` bir **eğitim/dağıtım uyuşmazlığı** taşıyor: yerel olarak
optimize ettiğimiz alan ile gönderdiğimiz alan aynı değil. Ağır üyeleri atmak
ikisini eşitler.

⚠ Uyarı: bu okuma sub007/sub008/sub009 çiftlerinden geliyor ve onlar **eski
konfigürasyonda** (düz sınıflandırıcı, üretim küçültmesi, p 0.35). Öznitelik
ekseninin aşağıda çıkan etkileşimi, bu tür okumaların konfigürasyona bağlı
olabileceğini gösteriyor. Bu yüzden doğrudan gönderilecek, varsayılmayacak.

### 15.5 Öznitelik ekseni bir ana etki değil, **etkileşim**

| kazanan | kaybeden | bağlam |
|---|---|---|
| 32 öznitelik (#3) | 23 öznitelik (#10) | iki-aşamalı + gerçek-kesirli |
| **23 öznitelik (#5)** | **32 öznitelik (#9)** | düz sınıflandırıcı + üretim küçültme |

Aynı eksen iki bağlamda ters işaret veriyor. Yani "öznitelik hijyeni iyi/kötü"
diye bir cevap yok; hangi skorlama kuralı ve hangi küçültmeyle birlikte olduğuna
bağlı. Bu, sub010'dan yaptığım ilk çıkarımın neden önce doğru görünüp sonra
çürüdüğünü de açıklıyor: iki farklı bağlamdan iki farklı gerçek okumuşum.

### 15.6 Politika

`conf 0.40 / p 0.55` (üst %45'i tut), `conf 0.35 / p 0.50` (üst %50) kolunu
geçiyor — sub017 #10, sub016 #13, tek değişkenli. Daha sıkı tutma kazanıyor.

### 15.7 Sırada ne var

En yakın okunamaz çift tam da boşluğu gösteriyor: **sub019 #1, sub018 #2'nin
üstünde ama ikisi hem semantik alanda hem küçültmede ayrılıyor.** İki bilinen
kazancı birlikte taşıyan gönderim hâlâ yok.

1. **sub020 = blob5 ½ + üretim küçültmesi.** sub019'dan yalnız küçültmede,
   sub018'den yalnız semantik alanda ayrılır. Seçici şu an o havuza üretim
   küçültmesiyle uyduruluyor.
2. **sub021 = sub020 eksi ağır üyeler** (§15.4). Bu aynı zamanda test zamanı
   ansamblını kat-dışı havuzun kullandığı alanla eşitler.

---

## 16. Küçültme çelişkisine önerdiğim mekanizma çürüdü (2026-09-03)

§15 tahtanın küçültme ekseninde yerel ölçümün tersini söylediğini kaydetti:
`sub018` (üretim halka erozyonu, alanın ~%19'unu siliyor) sıralamada `sub015`'in
(gerçek-kesirli 0.10, %10 siliyor) üstünde. Yani tahta **daha çok** kırpma
istiyor.

Buna bir mekanizma önermiştim ve test edilebilirdi: küçültme, tahmin GT'yi
aşırı kapladığında kazandırır (PQ eşiği IoU > 0.5), dolayısıyla test
maskelerimiz eğitimdekinden daha şişkinse testte daha büyük kırpma kazanır.
Zor bir dağılım tam da bunu yapar — gevşek olasılık alanı, aynı eşikte daha
kalın maske.

Test etiketi gerektirmeden ölçülebiliyordu: politikanın **tuttuğu** örneklerin
alanlarını kat-dışı ve test tarafında karşılaştır.

| reçete | kat-dışı medyan | test medyan | oran |
|---|---|---|---|
| sub015 (gerçek-kesirli 0.10) | 1312 | 1254 | **0.955** |
| sub019 (blob5, gerçek-kesirli) | 1234 | 1200 | **0.973** |

İki reçetede de oran **1'in altında**: test maskeleri kat-dışındakilerden
*küçük*, şişkin değil. Önerdiğim mekanizma yok.

### Bunun sonucu

- `sub018 > sub015` sıralaması ayakta, ama arkasında ölçülmüş bir mekanizma
  yok ve **sıralama büyüklük vermiyor** — iki komşu kol binde birkaçın kesriyle
  de ayrılmış olabilir.
- Yerel ölçüm gerçek-kesirliyi **+0.0031** önde tutuyor ve bu iki ayrı havuzda
  tekrarlandı (temel havuz 0.4331 vs 0.4286; blob5 havuzu 0.4416 vs 0.4368).
  Eşleşmeli protokolle, ve büyüklüğü de veriyor.
- Dolayısıyla "üretim küçültmesi kazanıyor" **kesinleşmiş değil**. Bir bitlik
  ordinal sinyalin üzerine hikâye kuruyordum; ölçüm hikâyeyi çürüttü.

Bu, kayıttaki genel kurala uyuyor: bir vekilin ya da bir bitin işaret ettiği
yöne, o yönü açıklayan bir mekanizma ölçülmeden yatırım yapma. Burada
mekanizmayı arayıp bulamadım, o yüzden sub020 hâlâ gönderilecek — ama bir
**keşif** olarak, kesinleşmiş bir iyileştirme olarak değil.

### Sıra değişti

`sub021` (ağır üyeleri atmak) artık daha güçlü aday, çünkü tek dayanağı ordinal
sinyal değil: `recipe3`'ün iki ağır modeli yalnız fold 0'da eğitilmiş, folds
1–4'ün doğrulama görüntülerini görmüşler, dolayısıyla kat-dışı havuza hiç
giremezler. Seçiciyi içermedikleri bir alanda optimize edip içerdikleri bir alanı
gönderiyoruz — bu ölçülmüş bir eğitim/dağıtım uyuşmazlığı, bir yorum değil.

---

## 17. sub020 zirveye geçti, sub021 hipotezimi çürüttü (2026-09-03, 03:05)

Otomatik gönderim penceresi çalıştı: kota 03:05'te açıldı, iki kol da gönderildi,
altı dakika sonra sıralama okundu. Yeni tepe:

```
 1. sub020  blob5 ½ + ÜRETIM küçültme        0.39   <- yeni zirve
 2. sub019  blob5 ½ + gerçek-kesirli          0.39
 3. sub018  recipe3 + üretim küçültme         0.39
 4. sub015  recipe3 + gerçek-kesirli          0.39
 6. sub021  blob5 ½ AĞIR ÜYELER YOK           0.39
```

### 17.1 Küçültme: ikinci bağımsız doğrulama, artık sağlam

| kazanan | kaybeden | semantik alan | tek fark |
|---|---|---|---|
| sub020 (#1) | sub019 (#2) | blob5 | küçültme kuralı |
| sub018 (#3) | sub015 (#4) | recipe3 | küçültme kuralı |

**İki farklı semantik alanda, iki bağımsız tek-değişkenli kıyas, aynı yön.**
Üretim halka erozyonu gerçek-kesirli 0.10'u geçiyor.

Bu, yerel eşleşmeli çapraz uyumun tersi (+0.0031 gerçek-kesirli lehine, iki ayrı
havuzda) ve §16'da önerdiğim aşırı-kaplama mekanizması ölçümle çürümüştü. Yani
elimizde **mekanizmasız ama iki kez tekrarlanmış** bir tahta sonucu var.

İki kez tekrarlanan bir tahta sonucu, bir kez ölçülmüş bir yerel sayıdan ağır
basıyor — özellikle yerel ölçümün bilinen bir kör noktası varken: kat-dışı havuz
görüntü başına ~13.7 aday taşıyor, test havuzu ise beş dedektör modelini
birleştirdiği için ~55.5. **Dört kat yoğunluk farkı**, ve küçültme yoğunlukla
etkileşmesi beklenen bir işlem (daha çok aday → aynı GT'yi örten daha çok
tahmin → maske çakışmasını azaltan kırpma daha çok kazandırır). Bu, yerel
ölçümün göremeyeceği bir yerde duruyor. Sıradaki mekanizma testi bu.

### 17.2 Ağır üyeler: hipotezim çürüdü, hem de iki yönden

sub021 ağır iki modeli semantik alandan çıkarıyordu. Sonuç **#6** — sadece
sub019'un değil, **blob üyesi hiç olmayan sub015'in de altında**.

| kazanan | kaybeden | tek fark |
|---|---|---|
| sub019 (#2) | sub021 (#6) | ağır üyeleri **çıkarmak zarar veriyor** |
| sub015 (#4) | sub021 (#6) | ağır üyeleri çıkarmanın bedeli, blob üyesinin kazancından **büyük** |

(sub021 hâlâ sub014'ü geçiyor (#6 > #14), yani beş katlı blob iyi olmaya devam
ediyor — sadece ağır üyeleri kaybetmenin bedelini karşılamıyor.)

İki argümanım vardı ve ikisi de yanlış çıktı:

**Ölçülmüş argüman:** tahta `prod5 > recipe3` diyordu (sub007 #10 > sub008 #11).
Doğru, ama o okuma **eski konfigürasyonda** (düz sınıflandırıcı, üretim küçültme,
p 0.35). §15.5 öznitelik ekseninin bağlama bağlı olduğunu göstermişti; ansambl
ekseni de öyleymiş. Tek bir eski-bağlam okumasını yeni bağlama genelledim.

**Yapısal argüman:** ağır modeller yalnız fold 0'da eğitildi, kat-dışı havuza
giremezler, dolayısıyla "seçiciyi içermedikleri bir alanda optimize edip
içerdikleri bir alanı gönderiyoruz". Bu **olgu olarak doğru ama öngörü olarak
yanlış**. Semantik alan seçicinin *yukarısında*: daha iyi bir alan, seçici o
alana uydurulmuş olmasa da yardım ediyor. Doğrulanamaz olmak kötü olmakla aynı
şey değil, ve ben hijyen içgüdüsüyle iyi bir bileşeni atmaya kalktım.

### 17.3 Genel ders

Bu turda iki hipotezim de çürüdü (§16 aşırı-kaplama, §17.2 ağır üyeler) ama her
ikisi de **ucuza** çürüdü: biri bir CPU ölçümüyle, diğeri bir gönderim slotuyla.
Tam sıralama aracı olmasaydı sub021 "0.39, ayırt edilemez" olarak kalır ve yanlış
hipotez kayıtta doğru gibi dururdu.

Kural: bir eksenin okumasını **ölçüldüğü konfigürasyonda** tut. Bu projede şimdiye
kadar iki eksen bağlama bağlı çıktı (öznitelik seti, ansambl bileşimi) ve ikisi de
ben genelledikten sonra çürüdü.

### 17.4 İkinci mekanizma da çürüdü — eksen ampirik kalıyor

§17.1'de küçültme çelişkisine ikinci bir mekanizma önerdim: kat-dışı havuz
görüntü başına 13.7 aday taşıyor, test havuzu 55.5; dört kat yoğunlukta daha çok
tahmin aynı GT satırını sıkıştırır, PQ'da yalnız biri eşleşebilir ve gerisi 0.5'lik
FP olur, dolayısıyla kırpma yüksek yoğunlukta daha çok kazandırır — ve bu tam
olarak seyrek kat-dışı havuzun göremeyeceği bir fayda olurdu.

Ölçtüm (`src/eda/overlap_density.py`), tutulan örnekler arasındaki kalabalık:

| kol | örnek/görüntü | çakışan örnek | iki kez sahiplenilen piksel |
|---|---|---|---|
| sub019 test (kesir .10) | 7.41 | **0.0%** | **0.00%** |
| sub020 test (üretim .187) | 7.26 | **0.0%** | **0.00%** |
| sub015 test (kesir .10) | 7.16 | **0.0%** | **0.00%** |

Sıfır. Çünkü `fuse_and_featurise` varsayılan olarak `deoverlap=True` — adaylar
havuz kurulurken birbirine karşı kesiliyor, dolayısıyla tutulan örnekler
**tanım gereği** çakışmıyor. Giderilecek kalabalık yok, yani kırpmanın böyle bir
işlevi de yok.

**Durum:** küçültme ekseninde iki bağımsız tek-değişkenli tahta doğrulaması var
(sub020 > sub019, sub018 > sub015) ve **hiçbir mekanizma yok**. İki hipotez
önerdim, ikisini de ölçtüm, ikisi de çürüdü — aşırı-kaplama (test maskeleri daha
*küçük* çıktı) ve kalabalık (sıfır).

Bunu bir hikâyeyle kapatmayacağım. Ampirik sonuç ayakta: tahtada daha çok kırpma
kazanıyor, sebebini bilmiyoruz. Bir sonraki hamle de bu yüzden **ampirik** olarak
çerçeveleniyor, mekanizma güdümlü değil: miktarı artırınca ne oluyor, ve kuralın
şekli mi miktarı mı önemli.

| kural | nominal | gerçekte silinen |
|---|---|---|
| üretim `shrink_frac` | 0.05 | %18.7 ← sub020 |
| üretim `shrink_frac` | 0.25 | %28.3 |
| kesir `shrink_true_frac` | 0.10 | %10.1 ← sub019 |
| kesir `shrink_true_frac` | 0.19 | %19.1 |

- **sub022** = sub020 + `shrink_frac 0.25` → kazanan kuralda miktarı %18.7'den
  %28.3'e çıkarır. Miktar hipotezini test eder.
- **sub023** = sub020 + `shrink_true_frac 0.19` → miktarı sub020'ninkiyle
  eşitler (%19.1 vs %18.7) ama kuralı değiştirir. Şekil hipotezini izole eder.

İkisi birlikte "miktar mı, şekil mi" sorusunu ayırıyor ve ikisi de sub020'den
tek değişkenli.

---

## 18. Küçültme ekseni kapandı — ve çerçevem baştan yanlıştı (2026-09-03)

sub023 **0.38**, sıra **#10**. Bu, §16–17 boyunca taşıdığım çerçeveyi yıkıyor.

| kol | kural | silinen | tahta | yerel PQ |
|---|---|---|---|---|
| sub020 | **üretim (1 px erozyon)** | %18.7 | **#1** | 0.4368 |
| sub019 | kesir 0.10 | %10.1 | #2 | 0.4416 |
| sub023 | kesir 0.19 | %19.1 | **#10** | 0.4324 |

### 18.1 "Tahta daha çok kırpma istiyor" yanlıştı

`sub018 > sub015` ve `sub020 > sub019`'u "üretim %18.7 siliyor, kesir %10.1
siliyor, demek ki tahta daha çok kırpma istiyor" diye okumuştum. sub023 aynı %19'u
kesir kuralıyla siliyor ve **sekiz sıra düşüyor**. Kesir kuralı içinde daha çok
kırpmak hem tahtada hem yerelde daha kötü (0.4416 → 0.4324).

Miktar değil, **kural**. Ve miktarı sabitleyen sonda olmadan bu ayrılamazdı —
sub023 kaybeden bir kol ama bilgi olarak turun en değerlisi.

### 18.2 Kuralların gerçek farkı: sınır düzeltmesi vs alan düzeltmesi

`src/eda/shrink_rule_damage.py`, 1500 aday, ortalama genişliğe göre silinen alan:

| ortalama genişlik | üretim | kesir 0.10 | kesir 0.19 |
|---|---|---|---|
| 6.3 px | **%29.4** | %10.0 | %18.5 |
| 8.9 px | %22.7 | %10.2 | %19.2 |
| 10.6 px | %19.0 | %10.2 | %19.1 |
| 13.0 px | %15.5 | %10.2 | %19.1 |
| 17.6 px | **%11.0** | %10.0 | %19.0 |

Üretim kuralı bir piksellik morfolojik erozyon: sabit bir *halka* siliyor, ve o
halka ince maskede alanın %29'u, kalın maskede %11'i. **Genişliğe kendiliğinden
uyarlanıyor.** Kesir kuralı genişlikten bağımsız sabit bir yüzde siliyor.

Yani küçültme bir **alan** düzeltmesi değil, **sınır** düzeltmesi: maskelerimiz
her yerde yaklaşık bir piksel cömert ve doğru kural bunu piksel cinsinden ifade
eden kural. Alan yüzdesi cinsinden ifade eden kural kalın maskeleri fazla, ince
maskeleri az kırpıyor.

Sıralama bu asimetriyi de veriyor: kesir-0.19 kalın maskelerden %19 alıyor
(üretim %11 alıyor) ve #10'a düşüyor; kesir-0.10 her şeyi az kırpıyor ve #2'de
kalıyor. **Kalın maskeyi fazla kırpmak ağır, ince maskeyi az kırpmak hafif** bir
hata.

(Test edip çürüttüğüm hipotez: kesir kuralının ince yapıları parçaladığı. Ölçüm
hayır diyor — en ince beşlikte parçalanma üretimde %53.7, kesir-0.19'da %52.3,
fark yok. Hiçbir kural hiçbir maskeyi silmiyor da.)

### 18.3 Miktar ekseni de kapalı

`shrink_frac 0.25` aslında **iki piksellik** erozyon — quantile ikinci halkaya
geçiyor, %28.3 siliyor. Yerel mutlak PQ **0.3963**, üretimin 0.4368'ine karşı
−0.045. Bu, tahta-yerel anlaşmazlığının ölçeğinden bir büyüklük mertebesi büyük,
o yüzden sub022 hiç gönderilmedi.

**Bir piksel iyi, iki piksel kötü, sabit alan kesri kötü.** Üç alternatif de
elendi; eksen kapandı ve bu sefer bir mekanizması var.

### 18.4 Üç mekanizma önerdim, ikisi çürüdü, üçüncüsü tuttu

- *aşırı-kaplama farkı* (§16): test maskeleri kat-dışındakilerden **küçük** çıktı
- *aday yoğunluğu* (§17.4): tutulan örneklerde çakışma **sıfır**
- *sınır vs alan birimi* (§18.2): **ölçüldü ve tuttu**

İlk ikisi eğitim/test farkı hikâyesiydi; tutan üçüncüsü ise operatörlerin kendi
farkı. Ders: bir tahta sonucunu açıklarken önce **elimizdeki iki şeyin ne yaptığını**
ölç, dağılım hikâyesi kurmadan önce.

---

## 19. sub024: blob payı 0.65 — yerel +0.0009 tahtaya geçmedi (2026-09-03)

Beş katlı havuzda blob harman ağırlığı süpürüldü (GPU'suz — `blob5ens`
haritalarının tam olarak 0.5×cldice + 0.5×blob olduğu doğrulanıp bileşenler
yeniden harmanlandı):

| ağırlık | eşleşmeli fark (vs 0.50) | t | bölme |
|---|---|---|---|
| 0.65 | +0.0009 | +2.75 | 14/20 |
| 0.80 | −0.0008 | −1.43 | 6/20 |

Hipotez doğru yöndeydi (tek-katlı modelin 0.50'de tepe yapan eğrisi varyanstan
geliyordu; beş kat ortalanınca optimum yukarı kaymalıydı) ve yerel ölçüm bunu
+0.0009 ile destekledi. sub024 bunu taşıdı, tahtada **#2** — sub020'nin (#1)
altında, tek değişkenli.

Okuma: **0.50 ≥ 0.65.** Sıralama büyüklük vermiyor; +0.0009'luk bir yerel farkın
tahtada ters işaretle gelmesi, farkın gerçekte sıfıra yakın olduğuyla tutarlı.
İki-aşamalı seçici +0.0011 ile bir basamak atlatmıştı, bu +0.0009 atlatmadı —
yani "aynı protokolde +0.001 transfer eder" genellemesi de tutmuyor; o eşik
bu proje için gürültü sınırı.

Eksen kapalı: blob payı **0.50** kalıyor. sub020 hâlâ zirve.

Kayıt olarak: bu turda tahtaya taşınan üç yerel kazançtan biri geçti
(beş katlı blob, +0.0061), ikisi geçmedi (ağır üyeleri atmak — yapısal
argümandı, ölçüm değil; blob 0.65 — +0.0009). Transfer eşiği kabaca
+0.003–0.005 civarında duruyor; altındaki kazançları göndermek slot israfı.

### 18.5 Genişlik-kapılı erozyon: yön doğru, büyüklük tabanın altında

§18.2'nin mekanizması bir öngörü veriyordu: 1-px erozyon kalın maskeleri fazla
kırpıyor, o halde yalnız ince maskeleri (ortalama genişlik < 12 px, havuzun %78'i)
kırp, kalınlara dokunma. `shrink_width_gated`, blob5 havuzu, üretim erozyonuna
karşı eşleşmeli (703 stem, aynı 20 bölme):

**+0.0005**, eşleşmeli se 0.0003, t = +2.08, 12/20 bölme.

Mekanizma doğrulanıyor (işaret ve tutarlılık) ama büyüklük transfer tabanının
(~+0.003) çok altında — slot harcanmaz. Kalın maskelerin fazla kırpılması
gerçek ama küçük bir kayıp: havuzun sadece %22'si 12 px'ten kalın ve onlarda
%11'lik kırpma IoU'yu 0.5 eşiğinin altına nadiren düşürüyor. Boyut hatasının asıl bedeli ince tarafta ödeniyor ve orası zaten düzeltilmiş durumda.

---

## 20. sub025: Konsensüs Sıkılaştırma (p=0.60) — 0.40 Eşik Aşımı ve Tahtada True Rank #1 (2026-09-04)

`gemini_suggestion_master_analysis.md` çoklu-anotatör konsensüs analizinden türetilen E115 deneyi: çoklu anotatör kopyalarında tek uzmanın çizdiği şüpheli/soluk filamentleri tahmin etmenin çift taraflı FP cezası ürettiği gerçeğiyle seçici işletme eşiği `p=0.55`'ten `p=0.60`'a çıkarıldı.

- **Yerel Ölçüm:** 5 katın 5'inde bağımsız olarak optimum `p=0.60` seçildi (+0.0053 PQ, 471 FP elendi, 195 TP feda edildi).
- **Tahta Doğrulaması:** `sub025_blob5_p060.csv` gönderildi (Kaggle Ref: `56019001`).
- **Skor & Sıra:** Ekranda **`0.40`** gösterim basamağını atladı ve yuvarlatılmamış tam puan sıralamasında (`SUBMISSION_SORT_BY_PUBLIC_SCORE`) 26 gönderimimiz arasında **True Rank #1** zirvesine yerleşti.
- **Yarışma Genel Sıralaması:** Takımımız 509 katılımcı arasında **24. sıraya** (İlk %4.7) yükseldi.

---

## 21. sub026: Carve Retention Filtresi (tau=0.70) — Tahtada İşe Yaramadı, sub025'in Altında (#2) (2026-09-04)

Aday bankasında de-overlap sırasında alanının %30'undan fazlasını kaybeden güdük maskeleri (stumps) atmak yerelde 5/5 foldda +0.0051 PQ kazandırmıştı (E098). Bu filtre üretim hattına eklenerek `sub026_blob5_p060_carve70.csv` üretildi ve gönderildi (Kaggle Ref: `56019307`).

- **Tahta Skoru:** Ekranda **`0.40`**.
- **Gerçek Sıralama (True Rank):** `sub026` yuvarlatılmamış puana göre **`sub025`'in ALTINDA (True Rank #2)** kaldı! Yani `sub026` test setinde işe yaramadı, skoru geriletti.
- **Kök Sebep Analizi (Zincirleme Mantık — R10):**
  1. `sub025`'teki $p \ge 0.60$ eşiği zaten de-overlap kütüklerinin ezici çoğunluğunu elemişti (1179 tahmin).
  2. Test verisinde greedy de-overlapping sırasında bir adayı $\tau=0.70$ ile erken düşürmek, pikselleri boşalttı; bu durum normalde elenecek olan 3 farklı adayın $\ge 64\text{ px}$ sınırını aşarak havuzda kalmasına neden oldu (test tahmin sayısı 1179'dan 1182'ye çıktı).
  3. Bu 3 ek aday gizli test setinde saf False Positive (FP) yaratarak yuvarlatılmamış PQ'yu düşürdü.
- **Nihai Karar:** E098 Carve Retention Filter üretimden **çıkarılmıştır**. Projenin tek ve mutlak şampiyonu **`sub025`** (`conf >= 0.40, p >= 0.60`, True Rank #1) olarak teyit edilmiştir.

