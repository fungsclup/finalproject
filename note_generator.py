import numpy as np
dizi = []
def converter(dosyaIsmi,show):
  array = []
  notes = [["do","C"] , ["fa","F"],
           ["la","A"], ["mi","E"],
           ["re","D"], ["si","B"],
           ["sol","G"]]
  ll = np.load(dosyaIsmi)
  for predict in ll:
    nota = predict[1]
    if(nota == "solanahtari"): continue
    for note in notes:
      if(note[0] in nota):
        yeniNota = nota.replace(note[0],note[1])
        uzunluk = yeniNota[1:]
        yeniNota = yeniNota[0]
        dosyaIsmi = predict[0]
        if(show == True):
          print("################\nDosya ismi : {} \nYeni Nota  : {} \nUzunluk : {} \n################\n".format(dosyaIsmi,yeniNota,uzunluk))
        array.append([dosyaIsmi,yeniNota,uzunluk])
  return array

dizi = converter("./autocrop.npy",False)
print(dizi)