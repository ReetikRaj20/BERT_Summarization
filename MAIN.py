from SUMMARIZE import *
# =============================================================================================================================

print("Enter your option:")
print("1. PDF        2. Youtube        3.Speech       4.Image     5.Text")
preference = int(input())

if preference == 1:
    print("Enter file name:")
    filename = input()

    myFile = open(r"C:/Users/Reetik Raj\OneDrive\Desktop\pdf\{file}.pdf".format(file=filename), "rb")
    output_file = open("out.txt", "w", encoding="utf-8")
    pdfReader = PyPDF2.PdfReader(myFile)
    numOfPages = len(pdfReader.pages)
    print("Reading pdf file.....")
    for i in range(numOfPages):
        page = pdfReader.pages[i]
        text = page.extract_text()
        output_file.write(text)
    output_file.close()
    myFile.close()
    f = open("out.txt", "r", encoding="utf-8")
    data = f.read()

    '''model = PunctuationModel()
    punct_text = data
    data = model.restore_punctuation(punct_text)'''

    print("Sending text for tokenization....")

    sentences = sent_tokenize(data)
    sum = bertSummarize(sentences)

    print()
    print("Saving summarized text....")
    f = open(r"D:\2023\Final year project\Code\summarizedtext\{file}.txt".format(file=filename), "w", encoding="utf-8")
    f.write(sum)

    f.close()
    f = open(r"D:\2023\Final year project\Code\summarizedtext\{file}.txt".format(file=filename), "r", encoding="utf-8")
    text = f.read()
    newtext = text.replace("\n", " ")
    print("Fine tuning the text....")

    f.close()
    f = open(r"D:\2023\Final year project\Code\summarizedtext\{file}.txt".format(file=filename), "w", encoding="utf-8")
    f.write(newtext)

    f.close()
    f = open(r"C:\Users\Reetik Raj\OneDrive\Desktop\pdf\{file}summarize.txt".format(file=filename), "w",encoding="utf-8")
    f.write(newtext)
    print("Successfully saved.")

# -----------------------------------------------------------------------------------------------------------------------

elif preference == 2:
    print("Enter Youtube Video ID:")
    id = input()

    data = yta.get_transcript(id)
    print("Extracting the text from the video...")
    transcript = " "
    for value in data:
        for txt, values in value.items():
            if txt == "text":
                transcript = transcript + " " + values
    print("sending text to restore punctuation....")
    model = PunctuationModel()
    punct_text = transcript
    result = model.restore_punctuation(punct_text)
    sentences = text_to_sentences(result).split(".")
    print("Sending in for tokenization.....")
    sum = bertSummarize(sentences)

    print("Saving into local system...")
    f = open(r"C:\Users\Reetik Raj\OneDrive\Desktop\pdf\{file}summarize.txt".format(file=id), "w", encoding="utf-8")
    f.write(sum)
    print("Successfully saved")

# -----------------------------------------------------------------------------------------------------------------------

elif preference == 3:
    r = sr.Recognizer()

    # Start the microphone
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 1
        print("Speak now:")
        audio = r.listen(source)
    print("Recognizing Now .....")
    try:
        print("You said")
        text = r.recognize_google(audio)
    except Exception as e:
        print("Error" + str(e))
    print(text)

    print()
    print("Sending to restore punctuation...")
    model = PunctuationModel()
    punct_text = text
    result = model.restore_punctuation(punct_text)

    sentences = sent_tokenize(result)
    print("Passing into BERT model for tokenization..")
    sum = bertSummarize(sentences)

    number = 1
    print("Saving file to local system....")
    f = open(r"C:\Users\Reetik Raj\OneDrive\Desktop\pdf\Lecture{number}summarize.txt".format(number=number), "w",encoding="utf-8")
    number = number + 1
    f.write(sum)
    print("Successfully saved.")

# ------------------------------------------------------------------------------------------------------------------

elif preference == 4:
    print("Enter Image name:")
    imagename = input()

    print("Reading image...")

    img = iio.imread(r"C:/Users/Reetik Raj\OneDrive\Desktop\pdf\{file}.JPEG".format(file=imagename))
    text = pytesseract.image_to_string(img)

    # Print the resulting text
    print("===========================================")

    print(text)

    print("Sent text into BERT model...")
    sentences = sent_tokenize(text)
    sum = bertSummarize(sentences)

    print("Saving file to local system....")
    f = open(r"C:\Users\Reetik Raj\OneDrive\Desktop\pdf\{imagename} summarize.txt".format(imagename=imagename), "w",encoding="utf-8")
    f.write(sum)
    print("Successfully saved.")

#-------------------------------------------------------------------------------------------------------------------------------------------
elif preference == 5:
    f = open("text.txt", "r", encoding="utf-8")
    data = f.read()
    data = data.splitlines()
    data = " ".join(data)

    print("Sending text for tokenization....")

    # Sentence Tokenization
    sentences = sent_tokenize(data)
    sum = bertSummarize(sentences)

    f = open(r"C:\Users\Reetik Raj\OneDrive\Desktop\pdf\summarizedtext\ManualText.txt", "w", encoding="utf-8")
    f.write(sum)
    print("Successfully saved")

    f.close()