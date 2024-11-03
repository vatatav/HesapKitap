from setuptools import setup, find_packages 
 
setup( 
    name="hesapkitap", 
    version="0.1.0", 
    packages=find_packages(), 
    install_requires=[ 
        'openai==1.0.0', 
        'pandas', 
        'PyMuPDF', 
    ], 
) 
