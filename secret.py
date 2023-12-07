import torchvision,torch,os,time,datetime
from torchvision.io import read_image

def encrypt(image:torch.Tensor,key:datetime.datetime):
    key = int(key.strftime("%Y%m%d%H%M%S"))
    torch.manual_seed(key)
    generator = torch.Generator().manual_seed(key)

    temp = torch.flatten(image)
    lenth = temp.numel()
    perm = torch.randperm(lenth,generator=generator)
    return temp[perm]

def decrypt(secret:torch.Tensor,key:datetime.datetime):
    key = int(key.strftime("%Y%m%d%H%M%S"))
    torch.manual_seed(key)
    generator = torch.Generator().manual_seed(key)
    
    lenth = secret.numel()
    perm = torch.randperm(lenth,generator=generator)
    perm = perm.sort().indices
    image = secret[perm]
    return image.reshape((1,220,200))

# def equal(img1:torch.Tensor,img2:torch.Tensor):
#     assert img1.numel()==img2.numel()
#     num = img1.numel()
#     if (img1 == img2).count_nonzero() == num:
#         return True
#     else:
#         return False

if __name__ == "__main__":
    img = read_image('/home/sunyf23/Work_station/PUF_Phenotype/Latency-DRAM-PUF-Dataset/grayscale_images/alpha/d/20/1.3/pat1/alpha_d_20_1.3_pat1_grayscale_5_.png')
    now = time.time()
    key = datetime.datetime.now()
    input = encrypt(img,key)
    print("\nImage encrypted.\n")
    output = decrypt(input,key)
    then = time.time()
    runtime = then - now
    print("Image decrypted.\n")
    print(f"--- En/Decryption completed in {runtime} seconds ---\n")
    
    if torch.equal(img,output):
        print("Image consistency passed.\n")
    else:
        print("Image consistency failed.\n")