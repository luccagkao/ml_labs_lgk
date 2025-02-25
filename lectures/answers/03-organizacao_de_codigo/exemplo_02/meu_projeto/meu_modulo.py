'''Módulo de exemplo para demonstrar a criação de pacotes.
'''
import colorama


def say_hello(name: str | None = 'world') -> None:
    '''Imprime uma saudação colorida na tela.

    Args:
        name (str, optional): Nome a ser saudado.
    '''
    # Prepara a Colorama para funcionar corretamente,
    # especialmente em ambientes Windows.
    colorama.init()

    styling = colorama.Fore.RED + colorama.Back.YELLOW + colorama.Style.BRIGHT
    reset_styling = colorama.Style.RESET_ALL

    print(styling + f'Hello, {name}' + reset_styling)

    colorama.deinit()
