from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from googletrans import Translator

translator = Translator()

finbert = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-esg", num_labels=4)
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-esg")
esg = pipeline("text-classification", model=finbert, tokenizer=tokenizer)

sentence = """//這61間食品公司合共市值達5442億元（以9月26日股價計算），是社會龍頭企業，其業務影響大量農民、工人生計，有責任保障工人權益，確保生產、供應鏈均無剝削工人。但調查結果反映食品企業缺乏披露環境及社會政策的意識，公眾無法監察其履行社會責任的表現。惟有港交所全面規管，強制要求企業匯報，才能有效增加企業透明度，改善其環境、社會政策。維他奶國際在眾企業中獲得較高分，因它曾提及致力實現聯合國「可持續發展目標」，包括消除飢餓、負責任地消費和生產 ，樂施會希望更多企業仿效此點，但強調整體表現仍屬參差。// 
=====Shared Post=====
 港上市公司社會責任表現 平均分僅10.6分 - 香港經濟日報 - iMoney - 政經 - 講金講心 香港作為國際金融中心，屢遭批評推行「環境、社會及管治（ESG）」步伐太慢，港交所去年起加強上市公司「ESG」披露資料責任"""

print(esg(translator.translate(sentence).text))
print(esg("I am hungry"))
