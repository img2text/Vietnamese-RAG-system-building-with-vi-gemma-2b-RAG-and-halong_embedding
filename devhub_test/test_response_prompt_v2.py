from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Khởi tạo tokenizer và model
print("Bắt đầu quá trình khởi tạo mô hình và tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("himmeow/vi-gemma-2b-RAG")
model = AutoModelForCausalLM.from_pretrained(
    "himmeow/vi-gemma-2b-RAG",
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Sử dụng precision thấp để tăng tốc (nếu có hỗ trợ)
)

print("Khởi tạo mô hình và tokenizer thành công!")

# Chuyển model sang GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

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
Độc lập - Tự do Hạnh phúc
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

# Mã hóa input text thành input ids
input_ids = tokenizer(input_text, return_tensors="pt")

# Chuyển input ids sang GPU nếu có
input_ids = input_ids.to(device)

# Thực hiện suy luận và tạo văn bản
with torch.no_grad():  # Ngừng tính toán gradient để tiết kiệm bộ nhớ
    outputs = model.generate(
        **input_ids,
        max_new_tokens=100,  # Giảm số lượng token sinh ra
        no_repeat_ngram_size=3,  # Giảm độ phức tạp mô hình
        temperature=0.7,  # Kiểm soát độ ngẫu nhiên (nếu do_sample=True)
        early_stopping=True,  # Dừng sớm khi đủ văn bản (chỉ áp dụng khi num_beams > 1)
        num_beams=3,  # Sử dụng beam search để áp dụng early_stopping
        do_sample=True,  # Bật chế độ sample (nếu muốn sử dụng temperature)
    )

# Giải mã và in kết quả
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# In kết quả trả về
print("Kết quả xử lý:")
print(decoded_output)
