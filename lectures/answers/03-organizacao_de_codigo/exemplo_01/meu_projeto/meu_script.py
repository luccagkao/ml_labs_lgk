'''Imprime uma saudação colorida na tela.
'''
import colorama


def main() -> None:
    '''Função principal do script.
    '''
    # Prepara a Colorama para funcionar corretamente,
    # especialmente em ambientes Windows.
    colorama.init()

    styling = colorama.Fore.RED + colorama.Back.YELLOW + colorama.Style.BRIGHT
    reset_styling = colorama.Style.RESET_ALL

    print(styling + 'Hello, World!' + reset_styling)

    colorama.deinit()


if __name__ == '__main__':
    main()
