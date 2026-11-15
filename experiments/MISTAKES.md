# Mistakes Log — aynı hataya iki kez düşme

Format: tarih | hata | görünen sebep | kök sebep | önlem

- 2026-08-19 | E011 ilk validation'da RAM OOM (3MB alokasyon bile başarısız) | ceiling script'i
  sanılan RAM baskısı | Kök sebep FARKLI: makinede BAŞKA bir projenin eğitimi çalışıyor (19.3GB,
  train_detector.py/biohub) + ultralytics'in 8 dataloader worker'ı Windows spawn'da her biri ~1GB
  | Önlem: (1) uzun eğitim başlatmadan önce `Get-Process python` ile makine paylaşımını kontrol et;
  (2) worker sayısını RAM bütçesine göre ayarla (--workers argümanı eklendi, 1536'da 2 kullan);
  (3) başkasının sürecini asla öldürme — önce CommandLine ile kimlik tespiti.

- 2026-08-19 | Dokümanlarda metrik çelişkisi (01: PQ, deep-research: Dice) haftalarca fark edilmeden
  kalabilirdi | iki doküman farklı tarihli kaynaklardan derlendi | metrik 7-9 Ağustos'ta değişti ve
  derlemeler farklı anları yakaladı; tek doğruluk kaynağı yoktu | Önlem: metrik/kural bilgisi TEK
  dosyada (docs/01) tutulur, her önemli karar önce oradan doğrulanır; kaynak = resmi scorer kodu.
- 2026-08-19 | Workflow'da `await parallel(...)` unutuldu → anında çökme | dikkatsizlik | JS'te
  Promise destructure sessizce tip hatası verir | Önlem: workflow şablonunda her parallel/pipeline
  çağrısının önünde await kontrolü.
- 2026-08-19 | Manifest ajanının disk fit'i fiziksel olarak imkânsız değerler üretti (R>1024) ve
  "outlier" etiketi tam ters | düşük çözünürlüklü threshold fit halo'yu diske dahil etti | fit
  doğrulanmadan downstream istatistik üretildi | Önlem: geometrik fit'ler görsel overlay ile
  doğrulanmadan downstream analiz yapılmaz (disk_refit.py protokolü); bağımsız iki yöntem çelişirse
  ikisi de şüpheli sayılır.
- 2026-08-19 | disk_refit v1'de "peak sonrası en dik düşüş" limb dedektörü, disk kadar parlak dış
  halkalı görüntülerde halkanın dış kenarına kilitlendi (R=1036, rms 0.47 — "mükemmel görünen yanlış
  fit") ve sahte site-bazlı plate-scale/merkez-kayması bulguları üretti | tek profilin fiziği
  incelenmeden dedektör tasarlandı | Gerçek: TÜM görüntülerde limb'de parlak rim tepesi var ve disk
  SABİT (R=900±0.7, merkez 1024±1; GONG remap) | Önlem: dedektör tasarımından önce ham profil örnekle;
  düşük-rms tek başına doğruluk kanıtı DEĞİL (yanlış yapıya mükemmel fit mümkün); sonuçlar fizik
  beklentisiyle çapraz kontrol edilir (plate scale ~1.078"/px × güneş yarıçapı ≈ 875-905px).
- 2026-08-19 | Topluluk bilgisi "139/180 test leak" EKSİKTİ — gerçek 180/180 | tek kernel'e
  güvenildi | kernel yazarı yalnız stem kesişimini kısmi listeyle yapmış olabilir | Önlem: kritik
  sayılar yerel olarak yeniden hesaplanır (bu projede yapıldı); kernel istatistiği asla direkt
  strateji girdisi olmaz.
- 2026-08-20 | E022'nin (mask_ratio=2) ultralytics val metrikleri E013'ten kötü göründü (Mask mAP50 0.596 vs 0.625) ve bir an için deney başarısız sanıldı | mAP düşük = model kötü varsayımı | Kök sebep: ultralytics segmentasyon validasyonu maskeleri AYNI mask_ratio ızgarasındaki GT'ye karşı ölçüyor. mask_ratio=4'te model bozuk hedefi üretir ve bozuk GT'ye iyi eşleşir; mask_ratio=2'de hedef keskinleşir, sınav zorlaşır. Yani metrik kendi bozukluğuyla ölçtüğü için sadakat kazancını GÖSTEREMEZ | Önlem: mask_ratio (veya hedef temsilini değiştiren herhangi bir ayar) karşılaştırmaları YALNIZCA harici exact-PQ scorer ile, gerçek 2048 GT'ye karşı yapılır. Framework'ün kendi val metriği bu eksende geçersizdir.
- 2026-08-21 | phys-kanal eğitimi epoch 6'da çöktü (AttributeError: 'FilamentSemantic' object has no attribute 'target') | kod hatası sanıldı | Kök sebep: eğitim KOŞARKEN train_semantic.py'yi düzenledim. Windows'ta DataLoader worker'ları spawn ile modülü YENİDEN import ediyor; yeni sınıf tanımı, eski pickle'lanmış dataset nesnesiyle uyuşmadı | Önlem: arka planda bir iş çalışırken onun import ettiği kaynak dosyaya DOKUNMA. Ya işi bitir, ya değişikliği ayrı bir dosyada yap, ya da num_workers=0 kullan. Uzun eğitim başlatmadan önce 'bu dosyayı düzenleyecek miyim' diye sor.
- 2026-08-22 | Uzun süreli eğitimi 'python ... | grep -E ... | head -6' ile başlattım; süreç ilk 6 satırdan sonra SIGPIPE ile öldü (exit 0, hiç epoch yok) | çıktıyı kısaltmak istedim | head/Select-Object -First boruyu kapatır, upstream süreç sinyal alır. Aynı hatayı daha önce 'Select-Object -First 12' ile de yapmıştım | Önlem: uzun işleri ASLA kesen filtreden geçirme (head/-First/tail -n ile canlı boru). Arka plana al, çıktıyı sonra dosyadan oku.
- 2026-08-22 | 'Hızlandırma' diye eklediğim channels_last eğitimi 4.7× YAVAŞLATTI (1339 vs 284 ms/adım) — üstelik GPU %99 göründüğü için ilk bakışta doğru çalışıyor sanılıyordu | channels_last AMP altında her zaman kazandırır varsayımı | Kök sebep: EfficientNet depthwise-separable convolution ağırlıklı; cuDNN'in channels_last depthwise kernel'leri bu konfigürasyonda kötü. channels_last standart convolution'lı ağlarda (ResNet vb.) kazandırır, depthwise ağırlıklılarda kaybettirebilir | Önlem: performans 'iyileştirmeleri' A/B ÖLÇÜLMEDEN benimsenmez; nvidia-smi utilization%% verimlilik göstergesi DEĞİL (yavaş kernel de %99 gösterir). Ölçüt: ms/adım.
- 2026-08-22 | Cache'li dataset Windows DataLoader worker'ları başlatılırken çöktü (OSError 22 / 'pickle data was truncated'); ayrıca worker başına 2.76 GB RSS görülmüştü | memmap ucuz referanstır sanıldı | Kök sebep: np.memmap PICKLE EDİLİRKEN tüm diziyi materyalize eder. Windows spawn dataset'i pickle'layıp boruya yazdığı için 1.7 GB'lık dizi boruyu taşırıyor ve her worker tam kopya taşıyor | Önlem: dataset'te memmap'i attribute olarak tutma — yolu sakla, worker içinde tembel aç, ayrıca __getstate__ ile pickle dışında bırak. Doğrulama: pickle 2358 MB → 9.5 MB.

## Uzun süren işleri çıktı filtrelerinden geçirmek (üçüncü tekrar)

Aynı kök, üç farklı kılık:

1. `... | grep ... | head -6` → `head` kapanınca SIGPIPE, **eğitim sıfır epoch'ta öldü**
   (çıkış kodu 0 olduğu için başarılı sanıldı). `Select-Object -First 12` ile de aynısı.
2. `python ... 2>&1 | grep -E "thr|standalone"` → script **çöktü ama traceback grep'e
   takıldı**, geriye yalnız sessizlik kaldı; hatayı ancak `mv`'nin başarısızlığından fark
   ettim.
3. `python -u ... | tr -d '\000' | tail -20` → `-u` verilse bile **`tr` ve `tail` blok
   tamponlar**; iş bitene kadar tek satır görünmez, ilerleme izlenemez.

Kural: arka planda koşan uzun işleri **filtreleyerek borulama**. Tam çıktıyı dosyaya bırak,
okurken filtrele. İlerleme gerekiyorsa işin kendisi bir `train.log` yazsın (artık
`train_semantic_arch.py` yazıyor) — dışarıdan tamponlanan bir borudan izlemeye çalışma.

## Refactor'ı içe aktarma testi yapmadan commit'lemek

`fuse_and_featurise` çıkarılırken eski döngünün beş satırı modül seviyesinde kaldı
(`return df` → SyntaxError). Modül o andan itibaren **hiç import edilemiyordu**, ama bunu
günler sonra `submit_v2` ilk kez koşturulunca gördüm. Bir dosyayı refactor edip
commit'lemeden önce en azından `python -c "import <modül>"` çalıştır.

## "Küçük fark" demeden önce kararı çevirip çevirmediğini ölç

Havuz float olasılıklardan, uygulama uint8 önbellekten featurise ediliyordu. Öznitelik
farkları ~0.002 görünüyordu ve "niceleme gürültüsü, önemsiz" diye geçiştirmeye hazırdım.
Ölçünce: selector olasılığında **max 0.549 fark, adayların %5.9'u tutma eşiğinin öbür
tarafına geçiyor**. Girdideki küçük fark çıktıda küçük kalmak zorunda değil — özellikle
ağaç tabanlı modelde eşik bölmeleri süreksiz. Ölçüt hep "karar değişiyor mu", "sayı yakın
mı" değil.

## Zincir betiğinin çocuğunu öldürmek zinciri durdurmaz

`aug_study.sh` üç kolu sırayla koşuyordu. Koşan python'ı `taskkill` ile öldürdüm — **bash
döngüsü bunu "bu kol bitti" sayıp bir sonrakini başlattı.** Ardından ebeveyn kabuğu
öldürünce döngü **öksüz kaldı ama koşmaya devam etti** ve kol açmayı sürdürdü. Sonuçta
kartta aynı anda **üç iş** oturuyordu; VRAM 8.6 GB'den 15.4 GB'ye çıktı ve her şey
yavaşladı. Belirtiyi ("neden 15.4 GB?") koşan işe değil, ölmüş sandığım işe bağlamak
yarım saatimi aldı.

Kural: zincirlenmiş bir işi durdurmak = **önce döngüyü koşan kabuğu, sonra python'ı, sonra
worker'ları** öldürmek — bu sırayla. Ve öldürdükten sonra `Get-CimInstance Win32_Process`
ile *komut satırına göre* filtreleyip gerçekten gittiğini doğrula; `nvidia-smi`'nin toplam
belleği tek başına yeterli kanıt değil ama iyi bir alarmdır.

## Windows PID'i ile bash PID'i aynı şey değil

Bekleyici betiğini `kill -0 56612` ile kurdum; 56612 `ps -W` çıktısındaki **WINPID**'di ve
bash'in `kill`'i onu tanımıyor → betik anında düştü. Windows'ta bir işin bitişini beklerken
PID yerine **işin kendi yazdığı günlükteki bir nişan** (`^done\.`) izlenmeli; hem PID
semantiğinden hem de "biten koşu `done` yazıyor ama `persistent_workers` yüzünden süreç
çıkmıyor" sorunundan bağımsız.

## Paralellik kartın boşluğu bittiğinde negatife döner

"GPU az kullanılıyor" geri bildirimi doğruydu ve paralel koşum 79% → 97% yapmıştı. Ama
16.3 GB'lik kartta 10 GB'lik bir ön-eğitimin yanına 7 GB'lik bir iş koyunca **ikisi de**
yavaşladı: ön-eğitim 345 → 488 s/epoch (+%41), çalışma 45 → ~450 s/epoch. Aynı iş sırayla
4.7 saat, paralel 11.3 saat. Paralellik boşluk varken bedava, boşluk bitince WDDM ana
belleğe taşıyor ve net zarar. Karar ölçüyle verilmeli: iki iş de yavaşlıyorsa serileştir.

## Git Bash'te olmayan bir komutla koşul yazmak — sessizce "hayır" döner

Uzun kolları ince ayarın arkasına zincirlemek için `while pgrep -f "..." ; do sleep 60; done`
yazdım. **Git Bash'te `pgrep` yok.** Komut bulunamadı → koşul yanlış → bekleme döngüsü
hiç dönmedi ve kol anında, ince ayarla **paralel** başladı; yani betiğin var olma sebebi
olan çekişmeyi betiğin kendisi yarattı. `set -u` bunu yakalamaz, çıkış kodu da 127 ile
sessizce geçer.

Kural: bir kabuk koşulunu bekleme mantığında kullanmadan önce **komutun var olduğunu
doğrula** (`command -v pgrep`). Windows'ta süreç beklemek için en güvenilir nişan hâlâ
**işin kendi yazdığı günlükteki bir satır** — hem `pgrep`/`kill -0` yokluğundan hem de
`persistent_workers` asılmasından bağımsız.

## Üç sayıya bakıp "eksen ölü" demek — trajektuara bak

Augmentasyon çalışması 30 epoch'ta light 0.6995 / heavy 0.6975 / medium 0.6960 verdi ve
ilk yorumum "fazla augmentasyon kazandırmadı" oldu. Trajektuar tersini söylüyordu: son
6 epoch eğimi light −0.00004 (bitmiş), medium +0.00041, heavy +0.00083 — augmentasyon
şiddetiyle **monoton**, ve heavy'nin en iyisi 29 epoch'un 28.'inde. Kol değerlendirilmedi,
yarıda kesildi. Yakınsamamış bir kol her zaman "kaybeden kol" gibi görünür.

Kural: bir ablasyonu ölü ilan etmeden önce **kolun bitip bitmediğine** bak — son epoch'ların
eğimi ve en iyi epoch'un yeri. Özellikle düzenleyici müdahalelerde (augmentasyon, dropout,
weight decay) yavaş yakınsama beklenen davranıştır, aynı bütçe adil bütçe değildir.

## Duman testi hata yakalamak içindir, karar vermek için değil

Ansambl ölçümünü 8 görüntüyle denedim (script çalışıyor mu diye) ve çıkan sıralamaya
bakıp "eğitim val'leri karşılaştırılamaz, `seed_b3_512` tuzağına ikinci kez düştük"
diye **kullanıcıya bildirdim**. 142 görüntünün tamamında sıralama eğitim val'leriyle
birebir tutuyordu (2048'de +0.0011 / −0.0005, eğitim val'inde +0.0010 / −0.0004).

Sekiz görüntü, 0.001'lik farkları ayırt etmek için gereken örneklem büyüklüğünün çok
altında; gördüğüm "ters dönme" saf gürültüydü. Kural: `--limit N` ile koşulan bir duman
testinin çıktısı **yalnızca "çöktü mü / makul aralıkta mı"** sorusunu cevaplar. Sayıya
bakıp yön değiştirmeden önce tam ölçümü bekle — özellikle o yön bir koşan işi
durduracaksa.

## Liderlik tablosunun ondalık çözünürlüğü, bir gönderimin ölçebileceğine taban koyar

sub008 (reçete-çeşitli ansambl, yerelde +0.0054 piksel-Dice) **0.38** aldı — sub007 ile
aynı. Beklenen PQ kazancı ~+0.003'tü ve tablo iki ondalık gösteriyor, yani "kazanç yok"
ile "+0.003 kazanç" **ayırt edilemez**. Gönderim harcandı, bilgi gelmedi.

Kural: bir gönderim ancak **≳0.005 PQ**'luk (≈0.009 piksel-Dice) bir değişimi ayırt
edebilir. Yerel tohum gürültümüz ~0.0015 olduğuna göre ölçülebilir bölge dar: bundan
küçük değişiklikleri yerelde 5-fold ile ölç, tabloya sorma. Gönderim hakkını yalnız
yapısal değişikliklere sakla.

## Kendi defterindeki en büyük kaldıracı okuyup yine küçüğünü optimize etmek

`docs/15` §6 haftalardır şunu yazıyordu: maske kalitesi **~0.19 PQ**, seçim boşluğu
~0.08 PQ. Ben seçim tarafında çalıştım — selector v5/v6, füzyon, küçültme, ansambl,
augmentasyon — ve toplamda ~0.01 topladım. Liderlik tablosunda dört takım yerel
çözünürlüğe geçip **+0.15** aldı, yani tam da o 0.19'luk kalemi.

Oracle'ımız (yerel 0.5078 ≈ LB 0.46) onların skorunun **altında**: mevcut aday
havuzumuzdan mükemmel seçim yapsak bile oraya ulaşamazdık. Bunu daha erken fark
edebilirdim — tavanın nerede olduğu ölçülmüştü.

Kural: "hangi eksende çalışıyorum ve o eksenin tavanı ne" sorusunu düzenli sor. Oracle
ölçümü zaten bunun için var; bir eksenin oracle'ı hedefin altındaysa **o eksen kapalıdır**,
ne kadar iyileştirirsen iyileştir.

## Zamanlama + büyüklük uyuşması nedensellik değildir — doğrudan test varken ona bak

Liderlik tablosu 0.40'tan 0.55'e sıçrayınca, aynı günlerde çıkan "blended-tile U-Net"
notebook'unu sebep saydım: tarih uyuyordu (08-22/08-24) ve büyüklük de uyuyordu (kendi
defterimizdeki ~0.19'luk maske-kalitesi kalemi). `docs/17`'yi yazdım, önbelleği kurdum,
4 saatlik eğitimi başlattım.

**Doğrudan test bir komut uzaktaydı ve onu atladım:** notebook'un yazarı tabloda nerede?
AnthonyTherrien **92. sırada, 0.35** — bizim 0.38'imizin altında. Teknik o skorları
üretmiyor.

Kural: bir tekniği "şu skoru açıklıyor" diye benimsemeden önce **onu kullananın skoruna
bak**. Halka açık bir yarışmada bu doğrudan gözlenebilir. Zamanlama uyuşması ve
büyüklük uyuşması ikisi de korelasyondur; yazarın sıralaması testtir.

İkinci ders: aynı gün bir public notebook'u "2. sıradaki takımın yöntemi" diye
inceledim. Notebook 08-19 tarihliydi, onların 0.54'ü ise 08-24'te geldi — yani baktığım
şey onların *eski* yaklaşımıydı. Üstelik bir başka takım (Varad) o notebook'u 23 saniye
sonra koşup **0.40** almıştı, yani notebook'un skoru zaten ölçülüydü. Bir yayının
tarihini o kişinin skorunun tarihiyle eşleştir.

## Ortalama katkıyı marjinal katkı sanmak — politikayı tersine çevirdi

"0.55 için ne gerekir" bütçesini yazarken bir kümenin değerini
`SQ·k / (k + (N−k)/2)` diye hesapladım ve şu sonuca vardım: *metrik, azınlık-konsensüslü
filamentleri (1/2, 1/3, 2/3) tahmin etmemeni söylüyor.* Bunu kullanıcıya da bildirdim.
**Yanlıştı.**

O formül kümenin **ortalama** katkısı. Marjinal katkı, tahmin etmeme durumuyla farktır ve
tahmin etmemek de bedava değil: o k satır FN olup paydaya `0.5k` ekliyor.

    ΔPay   = SQ·k
    ΔPayda = [k + 0.5(N−k)] − 0.5k = 0.5N
    marjinal = 2·SQ·k/N

PQ 0.41 ve SQ 0.706'da eşik `k/N > 0.29`; en düşük katman 1/3 = 0.333 bunu **geçiyor**.
Yani metrik hiçbir katmanı atlamamızı söylemiyor — tam tersine hepsini bulmamızı istiyor.
(Bu ayrıca `selector_v6`'nın `iou_sum/n > PQ/2` kuralının doğru olduğunu doğruluyor;
`iou_sum/n = SQ·k/N` olduğundan iki eşitsizlik aynı.)

Kural: bir eylemin değerini hesaplarken **karşı-olguyu da fiyatla**. "Bunu yaparsam ne
kazanırım" değil, "yapmakla yapmamak arasındaki fark ne". Metrikte bir şeyi atlamanın da
bir maliyeti varsa (burada FN), onu saymadan çıkan politika sistematik olarak fazla
muhafazakâr olur.

## Paylaşılan harness'ın çıktı yolları: varyant koşusu üretim dosyasının üstüne yazdı

`fused_shrink_eval`'e `--pool/--tag` eklerken tag'i yalnız sonuç JSON'una uyguladım;
`selector_v5_shrunk.joblib` satırı etiketsiz kaldı. Çok-eşik havuz denemesi koşunca
**üretim selector'ı** (gönderimin okuduğu dosya) deneysel havuzla eğitilmiş haliyle
değişti. Fark edilmeseydi bir sonraki gönderim sessizce yanlış selector'la çıkacaktı.

Kural: bir harness'a varyant parametresi eklerken, **yazdığı BÜTÜN dosyaları** tag'e
bağla — sonuç dosyası, model dosyası, önbellek. Tek birini unutmak, "varyantlar
birbirini ezmesin" garantisini en pahalı dosyada bozar. Ve varyant koşusundan sonra
üretim yolundaki her artefaktın kaynağını doğrula.

## Elindeki ölçüm sonucun tahminini zaten veriyorsa, deneyi koşmadan önce ona bak

Transdüktif öz-eğitim kulvarını kurdum (sözde-etiket önbelleği, karışık eğitim, iki kol,
sub009) ve LB 0.38 = 0.38 çıktı. Null'u sonradan açıklamak kolay oldu — ama aynı
açıklama deneyden ÖNCE de elimdeydi: `train_test_shift.py` train/test'in aynı havuzdan
i.i.d. bölündüğünü ölçmüştü (kovaryat kaymaları 0.03–0.15 sd, 31 ortak gün). Transdüktif
adaptasyonun kazancı dağılım kaymasından gelir; kayma yoksa adapte olunacak şey yoktur.

Kural: yeni bir deney tasarlarken önce "bu mekanizmanın kazanç şartı ne, ve o şartı
ölçen bir sonucum var mı?" diye sor. Mekanizmanın ön şartı ölçülmüş ve yoksa, deney
bir GPU günü değil bir paragraf olmalı.

## Ortam da bir deney değişkenidir: Smart App Control gece cv2'yi blokladı

29 Ağustos sabahı cv2 import'u "Uygulama Denetimi ilkesi bu dosyayı engelledi" ile
düşmeye başladı — 10 gündür sorunsuz çalışan aynı dosya. Teşhis: Windows Smart App
Control açık (VerifiedAndReputablePolicyState=1) ve bir itibar güncellemesi imzasız
cv2.pyd'yi karaliste etmiş. Çözüm invaziv olmayandı: taze PyPI wheel'i
(opencv-python 5.0.0) itibar kontrolünden geçti; SAC'ı kapatmak gerekmedi (o geri
alınamaz bir sistem kararı ve kullanıcıya sorulmadan yapılmamalı).

Yan etki: force-reinstall numpy'ı 2.5.2'ye çekti. Çekirdek yığın (torch/cv2/rle
roundtrip/ultralytics/cuda) duman testinden geçirildi. Kural: paket onarımından sonra
yalnız onarılan paketi değil, hattın uçtan uca duman testini koştur.

## "Insan tavani 0.84" — kendisiyle eslesmeyi anlasma sanmak

`public_gt_value.py`, halka acik MAGFiLO etiketlerini yarisma GT'sine karsi puanlayip
SQ **0.8393** verdi; ben bunu "iki insanin anlasma tavani" diye kaydettim (docs/18 §4,
rapor taslagi §1.2, iskelet, karakterizasyon). Yanlisti: halka acik dosyadaki **her
satirin GT'de birebir ikizi var**, yani olcum agirlikla *kendisiyle eslesme*; 1.0'dan
sapma da benim gonderim icin uyguladigim de-overlap bozulmasiydi.

Dogru olcum, farkli annotator'larin ayni filamenti cizdigi ciftler (2144 cift):
medyan maske IoU **0.629**, omurga seridi IoU **0.708**. Yani modelimizin SQ'su
(0.706) insan-insan maske anlasmasinin **ustunde** ve "SQ ekseninde 0.14 pay var"
diye yazdigim sey hic yokmus.

Kural: bir "anlasma" istatistigi hesaplamadan once karsilastirdigin iki kumenin
**bagimsiz** olup olmadigini dogrula. Ortak kaynaktan gelen iki dosya birbirini
olcmez, kendini olcer.

## Kendi iddiasini test etmeyen test

Scorer parite paketinin ilk surumu, IoU 0.665 ve 0.855'lik iki tahmini "esik alti" ve
"esik ustu" diye etiketliyordu. Ikisi de 0.5'in ustunde; yani dogrulamak icin yazdigim
**kati `> 0.5` semantigi hic calistirilmadi**. Parite "OK" verdi ama o ozelligi hic
olcmemisti.

Duzeltilmis paket artik IoU 0.4950 / **tam 0.5000** / 0.5050 uclusunu kosuyor ve tam
0.5'in eslesmedigini gosteriyor.

Kural: bir test durumunu etiketlemeden once **iddia ettigi ozelligi gercekten uretip
uretmedigini hesapla**. Yesil gecen ama yanlis seyi olcen bir test, testsizlikten
kotudur — guven verir, koruma vermez.

## Yogunlasma olcup kirlenme olcmemek — bozuk hedefle iki deney

Omurga hedefini `apply_aug`in ucuncu duzleminden gecirdim. O duzlem **agirlik haritasi**
icin tasarlanmis ve dondurmede kenari `borderValue=1.0` ile dolduruyor; agirlik haritasi
icin dogru (notr agirlik 1), omurga/iskelet hedefi icin felaket: 768lik bir kare 20 derece
dondurulunce kose bolgeleri buyuk ve hepsi "omurga" olarak isaretleniyor.

Olcum: kapsama (hedefin maske icinde kalan orani) **1.00 → 0.075**, hedefin kapladigi alan
**%0.11 → %6.38** (58 kat). Model, koselerin omurga oldugunu ogrenmeye calismis.

Kosmadan once bir kontrol yapmistim ama yanlis seyi olctum: *"maske ici enerji / genel
ortalama = 22.8x"*. Bu **yogunlasma**yi olcuyor ve kontaminasyon varken de yuksek cikiyor.
Dogru sorgu **kirlenme**ydi: hedefin ne kadari maskenin disinda?

Kural: bir hedefin dogrulugunu kontrol ederken "dogru yerde mi" degil **"yanlis yerde ne
kadar var"** diye sor. Ve bir yardimci duzlemi tasirken, o duzlemin **notr degerinin** ne
oldugunu (0 mi 1 mi) acikca belirle — varsayilan, onu ilk yazan kullanim icin dogrudur.

## 2026-09-10 — Kazanma stratejisi denetimi: düzeltilenler ve açık doğrulamalar

Ayrıntılar, kaynaklar ve deney sırası:
[strateji denetimi](../docs/30_WINNING_STRATEGY_AUDIT_2026-09-10.md).
Bu kayıt eski deney sonuçlarını yeniden ölçülmüş gibi değiştirmez.

### Genel dış veri izninden finalde ek girdi izni çıkarmak

Host cevabının ground-truth yasağı alınmış, hemen önceki H-alpha-only final cümlesi
atlanmıştı. Zincir: kısmi kaynak okuma → eğitim/çıkarım ayrımının kaybı → manyetik
kanallı final modelin uygun sanılması. `24ae785` kural belgesini düzeltti ve
`submit_v2` içine mag-checkpoint kontrolü ekledi; eski docs/29 yorumu geçerli değil.

Önlem: dış kaynağın **eğitimdeki** ve **final çıkarımındaki** rolünü ayrı kaydet;
genel izinle birlikte host'un özel sınırlamasını da oku. Kalan kontrol: final
reçetesinin tüm cache ve alternatif yollarla yalnız izinli girdilere ihtiyaç duyması.

### Yüklenmiş yardımcı verinin model tarafından kullanıldığını varsaymak

E162 manyetik veriyi yüklüyor ve dönüştürüyor, fakat cached dataset son tensörü üç
raw kanaldan kuruyordu. Checkpoint mag olarak etiketlenince inference farklı
kanallarla çalışacaktı. Zincir: benzer dataset sınıflarında yanlış noktaya yama →
yükleme koduna bakarak tamamlandı kabulü → etkisiz eğitim ve dağılım uyumsuzluğu.

`318d58c` düzeltti. İzole prob önce üç eşit kanal, sonra beklenen üç farklı kanal
verdi; iki sürümün hash'leri `outputs/research/winning_audit_2026-09-10*.json` içinde.
Önlem: değişkeni okumak yerine **modelin aldığı tensörü** doğrula; eğitim/inference
sırası, ölçek, eksik veri ve dönüşüm parity'si kontrol edilsin. Eski checkpoint
varsa kaynak sürümü bilinmeden yeni kodla devam etmek onarım değildir.

### GT evrenini aday havuzundan türetmek

Bazı değerlendirmeler `GT[stem in candidate_stems]` kullanıyor. Sıfır adaylı dört
training görüntüsü ve dört GT satırı FN hesabından çıkmış. Zincir: join kolaylığı →
aday yokluğunu görüntü yokluğu saymak → cezalandırılması gereken miss'in silinmesi.
Mevcut kayıt üzerinde cebirsel PQ iyimserliği yaklaşık +0.000120; küçük olması
varyantlar arası değerlendirme evreninin değişmesini meşru kılmaz.

Durum açık: benchmark evreni split manifestinden gelmeli. Aday üretilemeyen görüntü
sıfır tahminle puanlanmalı; eksik cache ise veri bütünlüğü hatası olarak ayrılmalı.

### Semantic component sonucunu bütün instance hedeflerine genellemek

E141'in hedef kimlikleri özgün polygon ID'lerinden değil, birleşik maskenin
connected component'larından geliyor. Zincir: semantik maskeye instance adı vermek
→ touching/split kimliklerini kaybetmek → dar negatif deneyle daha geniş aileyi
kapatmak. Önlem: polygon↔component uyuşmazlığını ölç; fark önemliyse yalnız hedef
kimliğini değiştirerek kontrollü deney yap. Negatif kaydın kapsamını açık yaz.

### Ampirik plato ve insan uyumunu matematiksel tavan saymak

Önceki “İnsan tavanı” kaydında kendisiyle eşleşme hatası doğru saptanmıştı; ancak
insan-insan medyan IoU ile eşleşmiş model tahminlerinin ortalama SQ'sı da aynı
istatistik değil. Bu kıyas veya tek anotatörde seçilen oracle'ın taşınmaması,
beklenen PQ'yu öğrenmenin bütün payını tükettiğimizi kanıtlamaz. Benzer biçimde
birkaç ensemble altkümesinin `1/k` uyumu bütün yeni bilgi kaynaklarını kapatmaz.

Önlem: sonucu ölçülen popülasyon, koşullanma, hedef, kapasite ve güven aralığıyla
sınırla. Mevcut GONG soft-mask damıtması da bütün MAE/DINO/SSL ailesini sınamaz;
düşük train/test shift, self-training'in tüm olası kazanç mekanizmalarını elemez.

### Kapanış kontrol listesi

- [x] Kural ve mag-tensör düzeltmeleri son kaynakta görüldü; ilk/son prob ayrıldı.
- [ ] Etkilenmiş uzak koşuların kaynak hash'i ve kanal parity'si doğrulandı.
- [ ] Bütün 707 görüntülü üretim-reçete benchmark'ı yeniden puanlandı.
- [ ] Checkpoint seçimi Dice yerine iç validasyondaki exact PQ ile karşılaştırıldı.
- [ ] E163 yardımcı hedef ile öğretmen damıtması ayrı sonuç/hipotez olarak kaydedildi.
- [ ] Negatif deneylerin kapsamı ve güncel submission/notebook eşleşmesi düzeltildi.

## 2026-09-11 — Semantik / instance / selector incelemesi

Ayrıntılar ve öncelikler:
[31_SEMANTIC_INSTANCE_SELECTOR_REVIEW](../docs/31_SEMANTIC_INSTANCE_SELECTOR_REVIEW_2026-09-11.md).
Bu kayıt yeni eğitim sonucu değil; `SI-AUDIT-00` kod ve artefakt denetimidir.

- **M2F çıkarımını standart uygulamayla aynı sanmak.** Yerel kod yalnız query sınıf
  olasılığını skor yapıyor; yazarların instance kodu maske içi olasılığı da kullanıyor.
  Ayrıca sigmoid→resize ile resize→sigmoid aynı maske değil. Zincir: basitleştirilmiş
  decoder → güven/geometri farkı → sonucu bütün mimariye atfetmek. Önlem: aynı
  checkpoint üzerinde skor ve resize sırasını ayrı ablation'larla sınamak. Gerçek
  PQ kazancı bu incelemede ölçülmedi.
- **Yerel IoU terimini final instance-IoU saymak.** 9×9 dilation bölgesi dışındaki
  400 sentetik FP piksel, ilgili kaybı hiç değiştirmedi. Toplam BCE/Dice bunları
  cezalandırır; bu yardımcı terimin kapsamı farklıdır. Polygon kimliğinin %99,5
  korunması bu ROI kapsamı konusunu çözmez. Önlem: hedefin hesaplandığı desteği ve
  finalde puanlanan maskeyi ayrı tanımlamak.
- **Bağlantılı kümeyi tek fiziksel instance garantisi saymak.** Güncel 10.042
  adayın 439'u içinde IoU≤0.5 üye çifti olan union-find kümelerinden geliyor.
  Bu sayı yanlış merge sayısı değildir. Önlem: zincir kümelerde gerçek kimlik kaybını
  ölçmeden genel split/merge politikası değiştirmemek.
- **Tekrarlı split sayısını bağımsız kanıt sayısı gibi kullanmak.** Aynı görüntüleri
  yeniden bölen 20 değerlendirmede `sd/sqrt(20)`, yeni veri genelleme belirsizliğini
  kendiliğinden vermez. Önlem: fiziksel grup üzerinden paired belirsizlik ve ayrıca
  dış fold/seed sonuçları; Monte Carlo kararlılığı ile veri belirsizliğini ayırmak.
- **İlk makul açıklamayı ölçmeden kabul etmek.** “128 px patch bütün ince yapıları
  kaybediyor” şüphesi geometrik kontrolde desteklenmedi: pozitiflerde tahmini medyan
  genişlik 10.75 px, 2 px altı %0.13. Önlem: açıklamanın gerekli ön koşulunu önce
  ölçmek; mevcut negatif deneyi sırf makul bir hikâye bulundu diye yeniden açmamak.

Durum: bulgular kaydedildi; M2F inference ablation'ları, normalize edilmiş koşullu
selector hedefi ve zincir-küme GT denetimi öneri olarak kaldı. Üretim kodu ve çalışan
eğitim kuyrukları bu incelemede değiştirilmedi.

## 2026-09-13 — a silent fallback reported a no-op as a null
`selector_feature_augmentation.py` guarded the augmented-training branch with
`if len(cols) == len(feats): concat(...)`. One feature (`log10_area`) was missing from the
jittered rows, so the branch never ran and the experiment compared the baseline against itself:
"+0.0000, 95% CI [+0.0000, +0.0000]". A zero-width confidence interval is the signature of a
comparison that did not happen, and it is indistinguishable from a real null to anyone reading
the number later. Rule: a guard that can silently change what an experiment measures must be an
assert, never an if.

## A string replacement that matched nothing, reported as success (2026-09-14)

Patching a figure script through a shell heredoc, two of four `str.replace` calls silently
matched zero times because the heredoc collapsed the backslashes in the anchor text. The script
printed "figure fixes applied" and the figure kept rendering the wrong denominator (a round 6600
instead of the metric's own 7367), overstating every mechanism's value by 12 %. It was caught
only by reading the regenerated file.

This is the same failure as the `if len(cols) == len(feats)` guard recorded above, in a different
costume: an operation that can silently do nothing, reporting the same output either way.

**Rule:** every programmatic edit asserts its anchor matched exactly the expected number of times,
and no patch script is trusted because it printed a success line. Where a shell heredoc mangles
the anchor, write the patch to a file instead of piping it.

## A cosmetic script that corrupted a measurement (2026-09-14)

Turkish writes the percent sign before the number, so a small script reordered all 77
percentages in the translated report. Its number pattern allowed a comma inside the digits,
which let one match run past a comma and swallow the value after it: the text

    "+0.0000, 95 % CI [+0.0000, +0.0000]"

became

    "+%0.0000,95 CI [+0.0000, +0.0000]"

destroying a confidence interval in a sentence that is itself about a measurement that did not
happen. Nothing about the rendered page looked wrong, and the script reported success.

It was caught by `scripts/check_translation.py`, which compares the multiset of numeric tokens
in the translation against the English original and refused the file. That check existed only
because translation was expected to be the risky step; the corruption came from our own
formatting pass afterwards.

Two rules, and the second is the general one:

**Rule: a pattern that rewrites text near a number must be written to exclude every separator
the number does not contain.** The fix was not to special-case the comma that broke; it was to
restrict the pattern to the exact shape a percentage has here, which removes the whole class.

**Rule: run the verification after the last thing that touches the file, not after the step you
thought was dangerous.** A cosmetic pass is still a write.

Also recorded: the same script was reverted mid-run by one of the translators, because the
instructions it had been given said to preserve percentages character for character. Two agents
editing one file from opposite rules leave it inconsistent and neither is wrong. The glossary
now states that percentages are reordered afterwards by a named script and must be left alone.
# 2026-09-16 — sub042 cascade training-band mismatch caught before submission

`submit_from_pool.cascade_admit` hard-coded the unbagged first-stage OOF scores.
Combining a bagged first stage with this default would train the cascade on a
different band population from the bagged OOF experiment. Added an explicit
`--cascade-scores` input with one-to-one join and missing-score validation.
For sub042 pass `outputs/eda/bag5_oof_scores.parquet`; retain the original default
for reproducing sub037. The fixed deployment quantile (0.85) must be measured
separately from the diagnostic that chooses a quantile inside each training fold.

## 2026-09-20 → 2026-09-24 — seven silent failures in one analysis week

Each of these produced output that looked like a result. None raised an exception at the point
where the error was made, which is the common thread and the reason they are recorded together.

### 1. A prediction id that matched no image, read as a measurement (09-20)
A shrink sweep wrote prediction ids `p_{i}`. `filament.pq.score` splits a prediction id on the
FIRST underscore and matches the prefix to the ground truth's image id, so every image matched
nothing and all seven arms scored PQ 0.0000. The zeros were caught only because they were
identical across arms. **Root cause:** the scorer's id contract is implicit (it lives in two
`split("_", n=1)` calls), not asserted. **Rule:** prediction ids are `{stem}_{k}`; any harness
that builds a prediction frame asserts that the set of prediction image ids intersects the GT
image ids before scoring.

### 2. The same shrink applied twice, nearly reported as "removing shrink gains +0.0487" (09-20)
Corrected sweep's "no shrink" row reproduced the atlas exactly (TP 4802, FN 3397) — because the
labelled pool's `rle` is ALREADY the deployed, shrunk mask (`_label_one_pool.py` calls
`label_pool(name, 0.05, ...)`). The sweep shrank it a second time. **Root cause:** the pool
column name `rle` does not say which object it is. **Rule:** before any mask operation on a pool,
reproduce the deployed PQ from it unchanged; if that matches, the pool is post-shrink.

### 3. A rule drawn from one example (09-20)
After sub047 lost on the board I wrote "a sign flip between local kept count and test row count
is a kill signal" into the ledger and into memory. Checked against every arm that has both
numbers, it failed: sub046 moved in the same direction and lost; sub041 did not move and ranked
#14. Retracted in ledger §40c. **Rule:** a proposed pre-submission gate is tested against every
past arm with the data before it is written anywhere as a rule.

### 4. A multivariate claim made from medians (09-22)
I said the frame-relative features "close the train/test distribution gap" because their
medians matched (det_conf percentile 0.533 vs 0.531). A classifier still separated local from
test candidates at AUC 0.952. **Rule:** "two distributions are the same" is established with a
multivariate two-sample test (a cross-fitted classifier AUC), never with marginal statistics.

### 5. A slice built in the wrong order, twice (09-22)
The per-quartile read indexed a covariate by the pool's 703 stems in global order, while
`selector_image_bagging.run` returns 707 folds_v1 stems concatenated in fold order. It crashed
with a length mismatch; the first "fix" still used the wrong order. **Rule:** reconstruct the
order from the harness's own code path and assert the length, as the final version does.

### 6. Three "it is running" messages about a file that did not parse (09-22 → 09-23)
A heredoc patch put a literal newline inside an f-string. The file had a syntax error, every
relaunch died instantly, and I read the previous run's log as if it were new. **Rule:** after
any code edit, `ast.parse` the file; after any launch, confirm the process exists and the log's
mtime is after the launch before reporting that anything runs.

### 7. An infinite loop from a replacement that contained its own search string (09-24)
`while old in s: s = s.replace(old, new, 1)` where `new` began with `old`. It never terminated;
the file was not damaged only because the write came after the loop. **Rule:** code is edited
with the Edit tool (exact, unique match, or `replace_all`), not with regex/`replace` loops in
heredocs. Four of the seven failures above came from heredoc patching.
*Correction (09-24, later):* the loop was not killed. The process (started 09-23 15:35) was still
running when found in the process list the next day, after about 27 CPU-hours and 870 MB of
growing string. Stopped then; `train_semantic.py` has no diff against the commit, so nothing was
written. The same day a command opened with `cat > file` without a heredoc and waited on stdin in
the background for ten minutes, read as "the scorer is slow". **Rule:** after stopping a process,
confirm by PID that it is gone; a background job with no output after a few minutes is checked in
the process list before it is waited on.

### Checklist added to every analysis turn
- [ ] prediction ids are `{stem}_{k}` and intersect GT image ids
- [ ] deployed PQ reproduced from the pool before any mask operation
- [ ] per-slice covariates ordered by the harness's own stem order, length asserted
- [ ] every edited file parses; every launched process is confirmed alive with a fresh log
- [ ] no rule written from fewer arms than the data allows testing it on
- [ ] "same distribution" claims backed by a classifier AUC, not medians

### 8. "Never measured", from a search in the wrong language (09-24)
I told the user, wrote into ledger §47/§49 and into memory that the `phys` input stack required by
AGENTS.md R8 had never been measured. It had: ledger line 112 records it (fold-0 pixel-Dice 0.6971
vs 0.7019) and report Appendix A.3 prints it. I had searched the Turkish ledger for English tokens
(`phys`, `local_contrast`, `channels`). Corrected in ledger §49a. **Rule:** before claiming an
experiment was never run, search the ledger in Turkish and English and by its expected result
value, and check Appendix A of the report.

### 9. The percentage reorder script breaks ranges (09-24)
`scripts/tr_percent_style.py` moves the percent sign in front of the LAST number only, so the
English "10–90 %" became "10–%90" instead of "%10–90" in the Turkish §4.7. Found by reading the
rendered section, not by the checker (the checker compares numbers without their sign, so it
cannot see this). Fixed by hand. **Rule:** after running the script, grep the Turkish files for
`[0-9]–%[0-9]` and correct each hit to `%a–b`; the script should learn ranges before it is relied
on again.

### 10. A "correction" planned from a thumbnail (09-24)
Reviewing the 380-pixel contact sheets, I decided that the unproposed dark structure in
20220714185352Th (ledger §51, finding 2) lay beyond the ±70° annotation scope and planned to
correct §51 to "not a loss". Measured on the 2048 image before writing, the largest dark component
has its centre at r 0.89, longitude 54°: inside scope, so §51 was right and the correction would
have introduced the error. A thumbnail at 1/5.4 scale cannot place a structure to ten degrees near
the limb, where longitude changes fastest with radius. **Rule:** a positional claim (longitude,
limb distance, overlap with a mask) is measured on the full-resolution frame before it is written
or used to overturn an earlier entry.

### 11. A noise floor taken from one repeat (09-25)
Ledger §49 recorded that the control "reproduced within 0.0006 with the same seed", and I set the
single-pair rule for trainer arms at ±0.01 Dice on that basis. Five identical controls later
spread 0.6437–0.6601 (sd 0.0059), so a pair difference has sd ≈ 0.008 and the band is ~1.2 sd.
hf-FiLM passed one pair at +0.014 and lost both replicates. The replicate step caught it before
any five-fold spend, which is why it was not a costly mistake, but the rule itself was
mis-calibrated. Two samples cannot estimate a spread. **Rule:** a noise floor is measured from at
least three repeats before a threshold is set on it. A trainer arm on the year holdout is read with
≥3 seed pairs. Epochs of one run are not independent samples.

### 12. A disc radius quoted for the wrong object, for a month (09-25)
Report §2.1 said the disc had median radius 991.7 px (914–1033). That is the equivalent-area radius
of the bright part of each frame from `src/eda/tile_geometry.py`, which includes the bright sky halo
around the limb in halo frames. The disc fitted to the limb (`disk_refit.py` v2, overlay-checked on
2026-08-19 and recorded in docs/02 §6) is 899.95 ± 0.73 px in every frame. The correct number had
been in the repository since the first week; the report took the by-product of another script.
Found when Talha asked whether the disc centre and diameter vary between frames. **Rule:** a
geometric quantity in the report is quoted from the measurement that was validated for it (docs/02),
never from a side output of a script built for something else.

### 13. A "live suspect" that the ledger had already closed (10-04)
Asked whether something set up wrongly could be holding recall down, I presented the uniform
one-pixel ring erosion as an untested suspect and started E215 (ring restored only on thin kept
masks). Ledger §39a already records width-gated erosion at +0.0005 (12/20 splits, below the floor)
and, more importantly, shows that the near-miss masks have the RIGHT width (ratio 1.04–1.06) and
are short (skeleton ratio 0.69–0.72). I had searched the ledger for "shrink", "min_area",
"retina", "NMS" and the report's English text, but not for the Turkish "erozyon"/"genişlik-kapılı".
Memory already carried this rule from §49a ("search the ledger in Turkish and by result value");
I applied it to other suspects and not to this one. The cost was one CPU hour, and the claim was
corrected in the same session before any decision rested on it. **Rule:** before calling an axis
untested, grep the ledger for the Turkish name of the operation AND its parameter values, and read
the section that owns the population (here §39–39b, the near-miss anatomy).

### 14. Three sweeps read against a reference built on a different field (found 10-04)
Ledger §31b/§32 and report §3.2/§3.4/A.8 said production was "a local optimum on every axis":
clustering, vote and NMS variants all lost 17–32 reachable rows. The sweep pools were built with
`--sem-tag _blob5ens`; the reference they were read against, `df_ref8_t020`, had been rebuilt for
E144 on the eight-arm committee field. Two variables changed in every comparison, and the 24-row
field difference was attributed to each threshold. Found when the E218 NMS 0.5 arm, which should
equal production, read −24. Against a control built through the sweep's own path, every axis is
flat within ±8 of 1510 rows, and the detector-geometry "net −17" is +7. No deployed setting
changes (flat axes), but four report passages stated a false mechanism. Root cause chain: the
reference pool was swapped project-wide when the deployment-faithful field was introduced; the
sweep scripts kept their own field; `pool_complementarity.py` accepts any two tags and never
checks that they differ in one setting; and no sweep contained a "production settings through
my own path" arm, which would have read −24 on day one. **Rule:** every sweep includes its own
production-settings rebuild as the first arm and is read against it; an arm that should equal the
control and does not invalidates the sweep.
