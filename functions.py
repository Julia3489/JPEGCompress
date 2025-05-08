from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from HA_tables import huffman_table_for_DC_Y
from HA_tables import huffman_table_for_DC_Cb_Cr
from huffman_tables_2 import huffman_table_for_AC_Y
from huffman_tables_2 import huffman_table_for_AC_Cb_Cr

def image_to_numpy_array(image_path):
    image = Image.open(image_path)
    color_mode = image.mode
    image_array = np.array(image)
    image_size = image.size  
    return image_array, color_mode, image_size

def RGB2YCbCr(RGB):
    height, width, _ = RGB.shape
    YCbCr = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            R = RGB[i, j, 0]
            G = RGB[i, j, 1]
            B = RGB[i, j, 2]
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
            Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
            YCbCr[i, j] = [Y, Cb, Cr]
    return YCbCr

def subsampling(I, n):
    X, Y = I.shape
    I_subsampled = I[::n, ::n]  
    return I_subsampled

def downsample_chrominance(ycbcr_image):
    y_channel = ycbcr_image[:, :, 0]
    cb_channel = ycbcr_image[:, :, 1]
    cr_channel = ycbcr_image[:, :, 2]

    cb_downsampled = subsampling(cb_channel, 2)
    cr_downsampled = subsampling(cr_channel, 2)
    y_downsampled = y_channel[::2, ::2]
    return y_downsampled, cb_downsampled, cr_downsampled

def split_into_blocks(channel, block_size=8, fill_value=0):
    h, w = channel.shape
    num_blocks_h = (h + block_size - 1) // block_size  
    num_blocks_w = (w + block_size - 1) // block_size  

    padded_h = num_blocks_h * block_size
    padded_w = num_blocks_w * block_size
    padded_channel = np.full((padded_h, padded_w), fill_value, dtype=channel.dtype)
    
    padded_channel[:h, :w] = channel
    blocks = []
    for i in range(0, padded_h, block_size):
        for j in range(0, padded_w, block_size):
            blocks.append(padded_channel[i:i+block_size, j:j+block_size])
    
    return blocks

def DCT(A):
    N, _ = A.shape
    D = np.zeros((N, N))
    for v in range(N):
        for u in range(N):
            S = 0
            for y in range(N):
                for x in range(N):
                    S += A[y, x] * np.cos(np.pi * v * (2 * y + 1) / (2 * N)) * np.cos(np.pi * u * (2 * x + 1) / (2 * N))
            if u == 0:
                Cu = 1 / np.sqrt(2)
            else:
                Cu = 1
            
            if v == 0:
                Cv = 1 / np.sqrt(2)
            else:
                Cv = 1
            
            D[v, u] = (2 / N) * Cu * Cv * S
    
    return D

def MQ_changes(c):
    MQ_Y = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109,103, 77],
        [24, 35, 55 ,64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    MQ_CbCr = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99] 
    ])

    if c <= 0:
        c = 1  

    if c < 50:
        S = int(5000 / c)
    else:
        S = int(200 - c * 2)
        
    MQ_Y_new = np.round((MQ_Y * S + 50) / 100).astype(int)
    MQ_CbCr_new = np.round((MQ_CbCr * S + 50) / 100).astype(int)

    MQ_Y_new[MQ_Y_new < 1] = 1
    MQ_Y_new[MQ_Y_new > 255] = 255
    MQ_CbCr_new[MQ_CbCr_new < 1] = 1
    MQ_CbCr_new[MQ_CbCr_new > 255] = 255

    return MQ_Y_new, MQ_CbCr_new

def quantize(M_dct, M_quantization):
    return np.round(M_dct / M_quantization).astype(int)

def zigzag(M):
    if M.size == 0 or M.shape[0] == 0:  
        return []
    N = len(M)
    result = []
    for d in range(0, N):
        if d % 2 == 0:  
            for j in range(0, d + 1):
                i = d - j
                result.append(int(M[i][j]))  
        else:  
            for i in range(0, d + 1):
                j = d - i
                result.append(int(M[i][j]))  
    for d in range(N, 2 * N - 1):
        if d % 2 == 0:  
            for j in range(d - N + 1, N):  
                i = d - j
                result.append(int(M[i][j]))  
        else:  
            for i in range(d - N + 1, N):  
                j = d - i
                result.append(int(M[i][j]))  
    return result

def differential_encoding(dc_coefficients):
    if not dc_coefficients:
        return []
    encoded = [dc_coefficients[0]]
    for i in range(1, len(dc_coefficients)):
        encoded.append(dc_coefficients[i] - dc_coefficients[i - 1])
    
    return encoded

def create_dc_ac_list(dc_encoded, zigzag_blocks):
    result = []
    
    for i in range(len(dc_encoded)):
        result.append(dc_encoded[i])
        ac_coefficients = zigzag_blocks[i][1:]  
        result.extend(ac_coefficients)  
    return result

def rle_encode(ac_coefficients):
    if not ac_coefficients:
        return []
    
    encoded = []
    count = 0

    for i, value in enumerate(ac_coefficients):
        if value == 0:
            if all(v == 0 for v in ac_coefficients[i:]):
                encoded.append((0, 0))
                return encoded
            count += 1
        else:
            if count == 0:
                encoded.append((0, value))
            else:
                if count > 0:
                    while count > 0:
                        if count > 15:
                            encoded.append((15, 0))  
                            count -= 16
                        else:
                            encoded.append((count, value))  
                            count = 0

    return encoded

def encode_dc_coefficients(dc_coefficients):
    encoded = []
    for coeff in dc_coefficients:
        if coeff == 0:
            encoded.append((2, '00'))  
        elif coeff < 0:
            abs_coeff = abs(coeff)
            binary_repr = bin(abs_coeff)[2:]  
            inverted_bits = ''.join('1' if bit == '0' else '0' for bit in binary_repr)  
            bit_count = len(inverted_bits) 
            encoded.append((bit_count + 1, '1' + inverted_bits))  
        else:
            binary_repr = bin(coeff)[2:]  
            bit_count = len(binary_repr)
            encoded.append((bit_count + 1, '0' + binary_repr))  

    return encoded

def encode_ac_coefficients(ac_coefficients):
    encoded = []
    for block in ac_coefficients:
        block_encoded = []  
        for coeff in block:
            zero_count = coeff[0] 
            number = coeff[1] 
            if number == 0:
                block_encoded.append((zero_count, 2, '00'))  
            elif number < 0:
                abs_number = abs(number)
                binary_repr = bin(abs_number)[2:]  
                inverted_bits = ''.join('1' if bit == '0' else '0' for bit in binary_repr)  
                bit_count = len(inverted_bits) 
                block_encoded.append((zero_count, bit_count + 1, '1' + inverted_bits))  
            else:
                binary_repr = bin(number)[2:]  
                bit_count = len(binary_repr)  
                block_encoded.append((zero_count, bit_count + 1, '0' + binary_repr))
        encoded.append(block_encoded)  
    return encoded

def huffman_encode_dc_y(dc_coefficients):
    encoded = []
    for coeff_tuple in dc_coefficients:
        category = coeff_tuple[0]  
        binary_repr = coeff_tuple[1]  
        try:
            category = int(category)  
        except ValueError:
            print(f"Неверное значение категории: {category}")
            continue  
        category = min(max(category, -11), 11)
        code_word, code_length = huffman_table_for_DC_Y[category]
        
        encoded.append((code_word, binary_repr))  
    
    return encoded

def huffman_encode_dc_cb_cr(dc_coefficients):
    encoded = []
    for coeff_tuple in dc_coefficients:
        category = coeff_tuple[0]  
        binary_repr = coeff_tuple[1]  
        try:
            category = int(category)  
        except ValueError:
            print(f"Неверное значение категории: {category}")
            continue  
        category = min(max(category, -11), 11)
        code_word, code_length = huffman_table_for_DC_Cb_Cr[category]
        encoded.append((code_word, binary_repr))  
    return encoded

def huffman_encode_ac_y(ac_coefficients):
    encoded = []
    for block in ac_coefficients:
        block_encoded = []  
        for coeff in block:
            zero_count = coeff[0]
            number = coeff[1]
            try:
                zero_count = int(zero_count)  
            except ValueError:
                print(f"Неверное значение категории: {zero_count}")
                continue  
            huffman_code_info = huffman_table_for_AC_Y.get((zero_count, number), None)
            if huffman_code_info is not None:
                huffman_code = huffman_code_info[0]
                code_length = huffman_code_info[1] 
                block_encoded.append((huffman_code, coeff[2]))  
            else:
                print(f"Код для ({zero_count}, {number}) не найден.")
        encoded.append(block_encoded)  
    return encoded

def huffman_encode_ac_cb_cr(ac_coefficients):
    encoded = []
    
    for block in ac_coefficients:
        block_encoded = []  
        for coeff in block:
            zero_count = coeff[0]
            number = coeff[1]
            try:
                zero_count = int(zero_count)  
            except ValueError:
                print(f"Неверное значение категории: {zero_count}")
                continue  
            huffman_code_info = huffman_table_for_AC_Cb_Cr.get((zero_count, number), None)
            if huffman_code_info is not None:
                huffman_code = huffman_code_info[0] 
                code_length = huffman_code_info[1] 
                block_encoded.append((huffman_code, coeff[2]))  
            else:
                print(f"Код для ({zero_count}, {number}) не найден.")
        encoded.append(block_encoded)  
    return encoded



def jpeg_compressor(image_path, quality=0):
    raw_image, color_mode, image_size = image_to_numpy_array(image_path)
    width, height = image_size
    image_size_2 = width/2, height/2 
    if color_mode == 'RGB':
        YCbCr = RGB2YCbCr(raw_image)
    elif color_mode == 'YCbCr':
        YCbCr = raw_image 
    elif color_mode == 'L':
        YCbCr = np.zeros((raw_image.shape[0], raw_image.shape[1], 3), dtype=np.float32)
        YCbCr[..., 0] = raw_image  
        YCbCr[..., 1] = 128          
        YCbCr[..., 2] = 128          
    elif color_mode == '1':
        YCbCr = np.zeros((raw_image.shape[0], raw_image.shape[1], 3), dtype=np.float32)
        YCbCr[..., 0] = raw_image * 255  
        YCbCr[..., 1] = 128              
        YCbCr[..., 2] = 128              

    else:
        raise ValueError("Неподдерживаемый цветовой режим: {}".format(color_mode))

    y_downsampled, cb_downsampled, cr_downsampled = downsample_chrominance(YCbCr)

    y_blocks = split_into_blocks(y_downsampled)
    cb_blocks = split_into_blocks(cb_downsampled)
    cr_blocks = split_into_blocks(cr_downsampled)

    dct_y_blocks = [DCT(block) for block in y_blocks]
    dct_cb_blocks = [DCT(block) for block in cb_blocks]
    dct_cr_blocks = [DCT(block) for block in cr_blocks]
    
    MQ_Y, MQ_CbCr = MQ_changes(quality)
    
    quantized_y_blocks = [quantize(dct_block, MQ_Y) for dct_block in dct_y_blocks]
    quantized_cb_blocks = [quantize(dct_block, MQ_CbCr) for dct_block in dct_cb_blocks]
    quantized_cr_blocks = [quantize(dct_block, MQ_CbCr) for dct_block in dct_cr_blocks]
    
    """for i, coded in enumerate(quantized_y_blocks[:3]):
        print(f"Закодированный блок {i+1}:") 
        print(coded)""" 
    
    zigzag_y = [zigzag(quantized_block) for quantized_block in quantized_y_blocks]
    zigzag_cb = [zigzag(quantized_block) for quantized_block in quantized_cb_blocks]
    zigzag_cr = [zigzag(quantized_block) for quantized_block in quantized_cr_blocks]
    
    #first_three_blocks = zigzag_y[:3]
    #print("Обход зигзагом первых трех блоков Y закодированное:", first_three_blocks)
    
    #dc_do_diff = [block[0] for block in zigzag_y]
    #print("DC коэффициенты Y до разностного кодирования:")
    #print(dc_do_diff)

    dc_y_diff = differential_encoding([block[0] for block in zigzag_y])  
    dc_cb_diff = differential_encoding([block[0] for block in zigzag_cb])  
    dc_cr_diff = differential_encoding([block[0] for block in zigzag_cr])  
    
    #print("DC коэффициенты Y после разностного кодирования:")
    #print(dc_y_diff)

    """print("DC коэффициенты Cb после разностного кодирования:")
    print(dc_cb_encoded)

    print("DC коэффициенты Cr после разностного кодирования:")
    print(dc_cr_encoded)"""
    
    encoded_dc_y = encode_dc_coefficients(dc_y_diff)
    encoded_dc_cb = encode_dc_coefficients(dc_cb_diff)
    encoded_dc_cr = encode_dc_coefficients(dc_cr_diff)
    
    #print("Обработанные DC коэффициенты Y:")
    #print(encoded_dc_y)

    """print("Обработанные DC коэффициенты Cb:")
    print(encoded_dc_cb)

    print("Обработанные DC коэффициенты Cr:")
    print(encoded_dc_cr)"""
    
    huff_dc_y = huffman_encode_dc_y(encoded_dc_y)
    huff_dc_cb = huffman_encode_dc_cb_cr(encoded_dc_cb)
    huff_dc_cr = huffman_encode_dc_cb_cr(encoded_dc_cr)
    
    #print("Хаффман DC коэффициенты Y:")
    #print(huff_dc_y)

    """print("Хаффман DC коэффициенты Cb:")
    print(huff_dc_cb)

    print("Хаффман DC коэффициенты Cr:")
    print(huff_dc_cr)"""
    
    ac_y_coefficients = [block[1:] for block in zigzag_y]  
    ac_cb_coefficients = [block[1:] for block in zigzag_cb]
    ac_cr_coefficients = [block[1:] for block in zigzag_cr]
    
    #print("AC Коэффициенты Y")
    #print(ac_y_coefficients)
    
    """print("AC Коэффициенты Cb")
    print(ac_cb_coefficients)
    
    print("AC Коэффициенты Cr")
    print(ac_cr_coefficients)"""
    
    rle_ac_y_encoded = [rle_encode(ac) for ac in ac_y_coefficients]
    rle_ac_cb_encoded = [rle_encode(ac) for ac in ac_cb_coefficients]
    rle_ac_cr_encoded = [rle_encode(ac) for ac in ac_cr_coefficients]
    
    #print("RLE AC Коэффициенты Y")
    #print(rle_ac_y_encoded)
    
    """print("RLE AC Коэффициенты Cb")
    print(rle_ac_cb_encoded)
    
    print("RLE AC Коэффициенты Cr")
    print(rle_ac_cr_encoded)"""
    
    encoded_ac_y = encode_ac_coefficients(rle_ac_y_encoded)
    encoded_ac_cb = encode_ac_coefficients(rle_ac_cb_encoded)
    encoded_ac_cr = encode_ac_coefficients(rle_ac_cr_encoded)
    
    #print("Форматед AC Коэффициенты Y")
    #print(encoded_ac_y)
    
    """print("Форматед AC Коэффициенты Cb")
    print(encoded_ac_cb)
    
    print("Форматед AC Коэффициенты Cr")
    print(encoded_ac_cr)"""
    
    huff_ac_y = huffman_encode_ac_y(encoded_ac_y)
    huff_ac_cb = huffman_encode_ac_cb_cr(encoded_ac_cb)
    huff_ac_cr = huffman_encode_ac_cb_cr(encoded_ac_cr)
    
    #print("Хаффман AC коэффициенты Y:")
    #print(huff_ac_y) 
    
    """print("Хаффман AC коэффициенты Cb:")
    print(huff_ac_cb) 
    
    print("Хаффман AC коэффициенты Cr:")
    print(huff_ac_cr)"""
   
    return huff_dc_y, huff_dc_cb, huff_dc_cr, huff_ac_y, huff_ac_cb, huff_ac_cr, MQ_Y, MQ_CbCr, image_size, image_size_2

def huffman_decode_dc_y(encoded_data):
    decoded = []
    
    for code_word, binary_repr in encoded_data:
        category = None
        for cat, (code, length) in huffman_table_for_DC_Y.items():
            if code == code_word:
                category = cat
                break
        
        if category is None:
            print(f"Не удалось найти категорию для кодового слова: {code_word}")
            continue
        decoded.append((category, binary_repr))
    
    return decoded

def huffman_decode_dc_cb_cr(encoded):
    decoded = []
    reverse_huffman_table = {v[0]: k for k, v in huffman_table_for_DC_Cb_Cr.items()}
    for code_word, binary_repr in encoded:
        category = reverse_huffman_table.get(code_word, None)
        if category is not None:
            decoded.append((category, binary_repr))
        else:
            print(f"Кодовое слово '{code_word}' не найдено в таблице.")
    return decoded

def huffman_decode_ac_y(encoded):
    decoded = []
    reverse_huffman_table = {v[0]: k for k, v in huffman_table_for_AC_Y.items()}
    for block in encoded:
        block_decoded = []
        for huffman_code, additional_info in block:
            zero_count_number = reverse_huffman_table.get(huffman_code, None)
            if zero_count_number is not None:
                zero_count, number_decoded = zero_count_number
                block_decoded.append((zero_count, number_decoded, additional_info))
            else:
                print(f"Кодовое слово '{huffman_code}' не найдено в таблице.")
        decoded.append(block_decoded)
    return decoded

def huffman_decode_ac_cb_cr(encoded):
    decoded = []
    reverse_huffman_table = {v[0]: k for k, v in huffman_table_for_AC_Cb_Cr.items()}
    for block in encoded:
        block_decoded = []
        for huffman_code, additional_info in block:
            zero_count_number = reverse_huffman_table.get(huffman_code, None)
            if zero_count_number is not None:
                zero_count, number_decoded = zero_count_number
                block_decoded.append((zero_count, number_decoded, additional_info))
            else:
                print(f"Кодовое слово '{huffman_code}' не найдено в таблице.")
        decoded.append(block_decoded)
    return decoded

def decode_dc_coefficients(encoded):
    decoded = []
    
    for bit_count, binary_repr in encoded:
        if binary_repr == '00':
            decoded.append(0)
        elif binary_repr[0] == '1':
            inverted_bits = binary_repr[1:]  
            original_bits = ''.join('0' if bit == '1' else '1' for bit in inverted_bits)
            abs_coeff = int(original_bits, 2)  
            decoded.append(-abs_coeff)  
        else:
            original_bits = binary_repr[1:]  
            coeff = int(original_bits, 2)  
            decoded.append(coeff)

    return decoded

def decode_ac_coefficients(encoded):
    decoded = []
    
    for block in encoded:
        block_decoded = []
        for zero_count, bit_count, binary_repr in block:
            if binary_repr == '00':
                block_decoded.append((zero_count, 0))
            elif binary_repr[0] == '1':
                inverted_bits = binary_repr[1:]  
                original_bits = ''.join('0' if bit == '1' else '1' for bit in inverted_bits)
                abs_number = int(original_bits, 2)  
                block_decoded.append((zero_count, -abs_number))  
            else:
                original_bits = binary_repr[1:]  
                number = int(original_bits, 2)  
                block_decoded.append((zero_count, number))

        decoded.append(block_decoded) 
    return decoded

def differential_decoding(encoded_coefficients):
    if not encoded_coefficients:
        return []
    decoded = [encoded_coefficients[0]]  
    for i in range(1, len(encoded_coefficients)):
        decoded.append(decoded[i - 1] + encoded_coefficients[i])  
    
    return decoded

def rle_decode(encoded_blocks):
    decoded_blocks = []
    for encoded in encoded_blocks:
        decoded = []
        for count, value in encoded:
            if count == 0 and value == 0:
                decoded.extend([0] * (63 - len(decoded)))  
                break
            if count > 0:
                decoded.extend([0] * count)
            decoded.append(value)
        if len(decoded) < 63:
            decoded.extend([0] * (63 - len(decoded)))  
        decoded_blocks.append(decoded)  
    return decoded_blocks

def add_dc_to_ac(dc_coefficients, ac_coefficients):
    if len(dc_coefficients) != len(ac_coefficients):
        raise ValueError("Количество DC коэффициентов должно соответствовать количеству блоков AC коэффициентов.")
    combined_blocks = []
    for dc, ac in zip(dc_coefficients, ac_coefficients):
        combined_blocks.append([dc] + ac)
    return combined_blocks

def zigzag_decode(zigzag_list, N):
    if len(zigzag_list) == 0:
        return np.zeros((N, N), dtype=int)
    M = np.zeros((N, N), dtype=int)
    index = 0
    for d in range(2 * N - 1):
        if d % 2 == 0:  
            for j in range(max(0, d - N + 1), min(d + 1, N)):
                i = d - j
                if index < len(zigzag_list):
                    M[i][j] = zigzag_list[index]
                    index += 1
        else:  
            for i in range(max(0, d - N + 1), min(d + 1, N)):
                j = d - i
                if index < len(zigzag_list):
                    M[i][j] = zigzag_list[index]
                    index += 1

    return M

def dequantize(M_quantized, M_quantization):
    return M_quantized * M_quantization 

def IDCT(D):
    N, _ = D.shape
    A = np.zeros((N, N))
    
    for y in range(N):
        for x in range(N):
            S = 0
            for v in range(N):
                for u in range(N):
                    if u == 0:
                        Cu = 1 / np.sqrt(2)
                    else:
                        Cu = 1
                    
                    if v == 0:
                        Cv = 1 / np.sqrt(2)
                    else:
                        Cv = 1
                    
                    S += Cu * Cv * D[v, u] * np.cos(np.pi * v * (2 * y + 1) / (2 * N)) * np.cos(np.pi * u * (2 * x + 1) / (2 * N))
            
            A[y, x] =  (2 / N) * S
    
    return A

def merge_blocks(blocks, channel_shape, block_size=8):
    h, w = channel_shape
    h = int(h)
    w = int(w)
    merged_channel = np.zeros((h, w), dtype=np.float32)
    num_blocks_h = (h + block_size - 1) // block_size
    num_blocks_w = (w + block_size - 1) // block_size
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = blocks[i * num_blocks_w + j]
            start_h = i * block_size
            start_w = j * block_size
            merged_channel[start_h:start_h + block.shape[0], start_w:start_w + block.shape[1]] = block
    return merged_channel[:h, :w]

def upsampling(I_subsampled, n):
    X_sub, Y_sub = I_subsampled.shape
    X = X_sub * n
    Y = Y_sub * n
    I_upsampled = np.zeros((X, Y), dtype=I_subsampled.dtype)

    I_upsampled[::n, ::n] = I_subsampled  
    for i in range(X):
        for j in range(Y):
            if I_upsampled[i, j] == 0:  
                nearest_i = (i // n) * n
                nearest_j = (j // n) * n
                I_upsampled[i, j] = I_upsampled[nearest_i, nearest_j]
    return I_upsampled

def upsample_chrominance(y_downsampled, cb_downsampled, cr_downsampled):
    h_y, w_y = y_downsampled.shape
    y_upsampled = upsampling(y_downsampled, 2)
    cb_upsampled = upsampling(cb_downsampled, 2)
    cr_upsampled = upsampling(cr_downsampled, 2)
    ycbcr_image_restored = np.zeros((h_y * 2, w_y * 2, 3), dtype=y_downsampled.dtype)
    
    ycbcr_image_restored[:, :, 0] = y_upsampled
    ycbcr_image_restored[:, :, 1] = cb_upsampled[:h_y * 2, :w_y * 2]
    ycbcr_image_restored[:, :, 2] = cr_upsampled[:h_y * 2, :w_y * 2]

    return ycbcr_image_restored

def YCbCr2RGB(YCbCr):
    height, width, _ = YCbCr.shape
    RGB = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            Y = YCbCr[i,j,0]
            Cb = YCbCr[i,j,1]
            Cr = YCbCr[i,j,2]
            R = Y + 1.402 * (Cr - 128)
            G = Y - 0.7141 * (Cr - 128) - 0.34414 * (Cb - 128)
            B = Y + 1.772 * (Cb - 128)
            RGB[i,j] = np.clip([R,G,B], 0, 255)
    return RGB

def decompress(huff_dc_y, huff_dc_cb, huff_dc_cr, huff_ac_y, huff_ac_cb, huff_ac_cr, MQ_Y, MQ_CbCr, image_size, image_size_2): 
    
    dehuff_dc_y = huffman_decode_dc_y(huff_dc_y)
    dehuff_dc_cb = huffman_decode_dc_cb_cr(huff_dc_cb)
    dehuff_dc_cr = huffman_decode_dc_cb_cr(huff_dc_cr)
    
    dehuff_ac_y = huffman_decode_ac_y(huff_ac_y)
    dehuff_ac_cb = huffman_decode_ac_cb_cr(huff_ac_cb)
    dehuff_ac_cr = huffman_decode_ac_cb_cr(huff_ac_cr) 
    
    decoded_dc_y = decode_dc_coefficients(dehuff_dc_y)
    decoded_dc_cb = decode_dc_coefficients(dehuff_dc_cb)
    decoded_dc_cr = decode_dc_coefficients(dehuff_dc_cr)
    
    decoded_ac_y = decode_ac_coefficients(dehuff_ac_y)
    decoded_ac_cb = decode_ac_coefficients(dehuff_ac_cb) 
    decoded_ac_cr = decode_ac_coefficients(dehuff_ac_cr) 
    
    dediff_dc_y = differential_decoding(decoded_dc_y)
    dediff_dc_cb = differential_decoding(decoded_dc_cb)
    dediff_dc_cr = differential_decoding(decoded_dc_cr)
    
    derle_ac_y = rle_decode(decoded_ac_y)
    derle_ac_cb = rle_decode(decoded_ac_cb)
    derle_ac_cr = rle_decode(decoded_ac_cr)
    
    zigzag_y = add_dc_to_ac(dediff_dc_y, derle_ac_y)
    zigzag_cb = add_dc_to_ac(dediff_dc_cb, derle_ac_cb)
    zigzag_cr = add_dc_to_ac(dediff_dc_cr, derle_ac_cr)
    
    quantized_y_blocks = [zigzag_decode(block[:8*8], 8) for block in zigzag_y]
    quantized_cb_blocks = [zigzag_decode(block[:8*8], 8) for block in zigzag_cb]
    quantized_cr_blocks = [zigzag_decode(block[:8*8], 8) for block in zigzag_cr] 
    
    dequantized_y_blocks = [dequantize(quantized_y_block.astype(float), MQ_Y) for quantized_y_block in quantized_y_blocks]
    dequantized_cb_blocks = [dequantize(quantized_cb_block.astype(float), MQ_CbCr) for quantized_cb_block in quantized_cb_blocks]
    dequantized_cr_blocks = [dequantize(quantized_cr_block.astype(float), MQ_CbCr) for quantized_cr_block in quantized_cr_blocks]
    
    y_blocks = [IDCT(block) for block in dequantized_y_blocks]
    cb_blocks = [IDCT(block) for block in dequantized_cb_blocks]
    cr_blocks = [IDCT(block) for block in dequantized_cr_blocks] 
    
    Y = merge_blocks(y_blocks, image_size_2, block_size=8)
    Cb = merge_blocks(cb_blocks, image_size_2, block_size=8)
    Cr = merge_blocks(cr_blocks, image_size_2, block_size=8)

    YCbCr = upsample_chrominance(Y, Cb, Cr) 
    RGB_arr = YCbCr2RGB(YCbCr)
    
    RGB_arr = RGB_arr.astype(np.uint8)  

    image = Image.fromarray(RGB_arr)

    image.save('output_IMG_RGB_0.png')
    
    

    
huff_dc_y, huff_dc_cb, huff_dc_cr, huff_ac_y, huff_ac_cb, huff_ac_cr, MQ_Y, MQ_CbCr, image_size, image_size_2 = jpeg_compressor("C:\\Users\\aleks\\OneDrive\\Рабочий стол\\lab2AISD\\Img.jpg", quality=0)
decompress(huff_dc_y, huff_dc_cb, huff_dc_cr, huff_ac_y, huff_ac_cb, huff_ac_cr, MQ_Y, MQ_CbCr, image_size, image_size_2)  
#decompress(packed_data)
#print('Функции выполнены')




    



