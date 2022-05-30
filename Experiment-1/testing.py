import tika
tika.initVM()
from tika import parser

parsed = parser.from_file("Data/Raw Data/smm29-Groups_13_and_14.pdf")
print(parsed["content"])