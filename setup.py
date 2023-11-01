from setuptools import find_packages,setup
from typing import List

HYOEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    this funtion will return the list of requirements
    '''
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements]

        if HYOEN_E_DOT in requirements:
            requirements.remove(HYOEN_E_DOT)

        return requirements



setup(
    name = 'mlproject',
    version = '1.2.0',
    author ='Akhil',
    author_email ='akhil96me@gmail.com',
    packages =find_packages(),
    install_requires= get_requirements('requirements.txt')

)