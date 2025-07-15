## Execution of bre for dependencies in Brefile
brew bundle --file ./Brefile

## Post execution
chsh -s $(which zsh)
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"