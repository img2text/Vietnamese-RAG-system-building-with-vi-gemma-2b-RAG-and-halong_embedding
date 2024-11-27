from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Log bắt đầu quá trình
print("Bắt đầu quá trình khởi tạo mô hình và tokenizer...")

# Khởi tạo tokenizer và model
try:
    tokenizer = AutoTokenizer.from_pretrained("himmeow/vi-gemma-2b-RAG")
    model = AutoModelForCausalLM.from_pretrained(
        "himmeow/vi-gemma-2b-RAG",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("Khởi tạo mô hình và tokenizer thành công!")
except Exception as e:
    print(f"Thất bại trong việc tải mô hình và tokenizer: {e}")
    exit()

# Chuyển model sang GPU nếu có
if torch.cuda.is_available():
    model.to("cuda")
    print("Mô hình đã chuyển sang GPU.")
else:
    print("Không có GPU, mô hình sẽ chạy trên CPU.")

# Định dạng prompt yêu cầu trả về JSON
prompt = """
### Instruction and Input:
Dựa vào ngữ cảnh/tài liệu sau:
{input_data}
Hãy trả lời câu hỏi: {query}

### Response (Dưới dạng JSON):
{{
    "full_name": "",
    "dob": "",
    "sex": "",
    "nationality": "",
    "place_of_origin": "",
    "place_of_residence": ""
}}
"""

# Dữ liệu đầu vào
input_data = """
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tư do Hanh phúc
SOCIALIST REPUBLIC OF VIET NAM
Independence - Freedom Happiness
CĂN CƯỚC CÔNG DÂN
Citizen Identity Card
Số ID: 252458456854
Họ và tên / Full name: LÊ VĂN VIỆT
Ngày sinh / Date of birth: 28/01/1978
Giới tính / Sex: Nam
Quốc tịch / Nationality: Việt Nam
Quê quán / Place of origin: Tam Tiến, Núi Thành, Quảng Nam
Nơi thường trú / Place of residence: Thôn Quảng Bính, Cang nghiệt, Nghĩa Thắng, Đắk Nông
"""

# Câu hỏi
query = "Hãy cho tôi các thông tin về dữ liệu người này?"

# Định dạng input text
input_text = prompt.format(input_data=input_data, query=query)
print("Đã định dạng input text.")

# Mã hóa input text thành input ids
try:
    input_ids = tokenizer(input_text, return_tensors="pt")
    print("Mã hóa input text thành công!")
except Exception as e:
    print(f"Thất bại trong việc mã hóa input text: {e}")
    exit()

# Chuyển input ids sang GPU nếu có
if torch.cuda.is_available():
    input_ids = input_ids.to("cuda")
    print("Input đã chuyển sang GPU.")
else:
    print("Input sẽ chạy trên CPU.")

# Tạo văn bản bằng model
try:
    outputs = model.generate(
        **input_ids,
        max_new_tokens=500,
        no_repeat_ngram_size=5,  # Ngăn chặn lặp lại các cụm từ 5 gram
    )
    print("Tạo văn bản thành công!")
except Exception as e:
    print(f"Thất bại trong việc tạo văn bản: {e}")
    exit()

# Giải mã và in kết quả
try:
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Giải mã kết quả thành công!")
except Exception as e:
    print(f"Thất bại trong việc giải mã: {e}")
    exit()

# In kết quả trả về
print("Kết quả trả về:")
print(decoded_output)
