	for ag in {'max','min','avg','std','med'}
	do
		for ap in {'max','min','avg','std','med'}
		do
			printf "ap-$ap ap-$ag \n"
			python3.6 ~/repos/IRQPP/correlation.py ~/baseline/UQV/map1000-$ag ~/baseline/UQV/map1000-$ap -t $1 | cut -d ':' -f 2
			echo
	done
done
