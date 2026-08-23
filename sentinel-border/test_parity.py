from backend.modules.ocr_engine import run_ocr
from backend.modules.validator import check_viz_mrz_parity
import sys

with open('C:/Users/sumit/.gemini/antigravity-ide/brain/0bdcd3ff-d47b-451e-9568-53b532fb4834/.user_uploaded/media_1787514093820.jpg', 'rb') as f:
    data = f.read()

result = run_ocr(data, 'image/jpeg')
print('VIZ TEXT:', result.viz_text)
print('MRZ SURNAME:', result.surname)
print('MRZ GIVEN_NAMES:', result.given_names)
print('MISMATCHES:', check_viz_mrz_parity(result)[1])
