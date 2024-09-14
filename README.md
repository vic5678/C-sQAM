# C-sQAM
C-sQAM is a circular QAM with spike elements. After the research I conducted for different constellations for SWIPT systems, based on rectangular spike QAM, I came up with the idea to design a circular QAM with spikes and investigate its behavior for SWIPT systems.
C-s-QAM is a circular QAM but with spike elements on the outermost circle. Let’s consider M symbols , N circles , n the elements per circle and m number of spikes. We
will construct R1 with the formula :R1 = dmin/(2sin( π/n )).
Then, we find R2 so that the distance between any symbol on R2 and its adjacent symbols at R1 be dmin, if possible or slightly larger. In the same way we construct the remaining consecutive
of the circles up to the outermost RN.
We calculate the energy of the existing symbols, paying attention to subtract the spike elements of the outermost circle and then we define the remaining energy that has to be covered
so as to keep Es stable. From this remaining energy we will define the distance of the spikes.
First of all, I constructed it, then I observed its PAPR-dmin behavior, which outperforms the simple CQAM and is the same with the spike-QAM and then I did mathematicals calculations to find a SEP formula which is confirmed by monte carlo simulation too.
Last but not least, I am currently changing the number of the symbols in each circle to observe its behavior and optimize it.
So, C-sQAM is an optimal constellation for SWIPT systems.
