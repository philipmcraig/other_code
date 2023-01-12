# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific aliases and functions

export PATH="$PATH:~/sublime_text_3/"

alias sci1="ssh -A -Y sci1.jasmin.ac.uk"
alias sci2="ssh -A -Y sci2.jasmin.ac.uk"
alias sci3="ssh -A -Y sci3.jasmin.ac.uk"
alias sci4="ssh -A -Y sci4.jasmin.ac.uk"
alias sci5="ssh -A -Y sci5.jasmin.ac.uk"
alias sci6="ssh -A -Y sci6.jasmin.ac.uk"
alias xfer1="ssh -A -Y xfer1.jasmin.ac.uk"
alias xfer2="ssh -A -Y xfer2.jasmin.ac.uk"
alias raindir="cd /badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/rainfall/day/v20181126/"
alias ncasdir="cd /gws/nopw/j04/ncas_climate_vol1/users/pmcraig"
alias canopy="~/Canopy/canopy"
alias canopy_terminal='source $HOME/.local/share/canopy/edm/envs/User/bin/activate'
alias ncas_sci1="ssh -A -X sci1.ncas-sci-m.jasmin.ac.uk"
alias cmipdir="cd /badc/cmip6/data/CMIP6/CMIP"

shopt -s cdable_vars
export PCNCAS=/gws/nopw/j04/ncas_climate_vol1/users/pmcraig
