# specify directory to install
cask_args appdir: "/Applications"

# install packages
tap "homebrew/bundle"
tap "homebrew/cask"
tap "homebrew/core"
tap "schniz/tap"
brew "caffeine"
brew "python"
brew "git"
brew "helm"
brew "jq"
brew "kubernetes-cli"
brew "yq"
brew "zsh"
brew "zsh-autosuggestions"
brew "zsh-completions"
brew "zsh-syntax-highlighting"

# Casks
cask "appcleaner"
cask "iterm2"
cask "visual-studio-code"
cask "google-chrome"
cask "spotify"
