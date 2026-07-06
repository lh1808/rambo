# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
#        . "/opt/conda/etc/profile.d/conda.sh"
#    else
#        export PATH="/opt/conda/bin:$PATH"
#    fi
#fi
#unset __conda_setup
## <<< conda initialize <<<

##activate environment
#conda activate generic



# --- Fix SSH key permissions ---
echo "--- Git Prepare ---"
chmod 600 ~/.ssh/id_rsa

# --- Git Credentials ---
whereis git
git --version
git config --global user.email "marc.langhauser@lan.huk-coburg.de"
git config --global user.name "Langhauser, Marc"

# --- Git Clone Repo ---
echo "--- Git Clone Repo ---"
pushd /home/ubuntu/
git clone ssh://tfs.lan.huk-coburg.de:22/web/DefaultCollection/GIT_Projects/_git/da-hf1-rubin
popd

# -- Install Environment With Pixi ---
echo "--- Init Pixi ---"
whereis pixi
pixi --version

# Go To Pixi .toml Dir 
pushd /home/ubuntu/da-hf1-rubin

# Install Env
pixi install
pixi info
pixi add pyreadstat
pixi add seaborn

# Run python script 
echo "--- Copy Python Script into current dir and run ---"
pixi run python /mnt/Production/PBV_production/Production_PBV_Brief_Rubin.py
echo "Finished PBV_Brief"
