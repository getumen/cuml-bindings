package cuml4go

type Metric int

const (
	/** evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk) */
	L2Expanded = 0
	/** same as above, but inside the epilogue, perform square root operation */
	L2SqrtExpanded = 1
	/** cosine distance */
	CosineExpanded = 2
	/** L1 distance */
	L1 = 3
	/** evaluate as dist_ij += (x_ik - y-jk)^2 */
	L2Unexpanded = 4
	/** same as above, but inside the epilogue, perform square root operation */
	L2SqrtUnexpanded = 5
	/** basic inner product **/
	InnerProduct = 6
	/** Chebyshev (Linf) distance **/
	Linf = 7
	/** Canberra distance **/
	Canberra = 8
	/** Generalized Minkowski distance **/
	LpUnexpanded = 9
	/** Correlation distance **/
	CorrelationExpanded = 10
	/** Jaccard distance **/
	JaccardExpanded = 11
	/** Hellinger distance **/
	HellingerExpanded = 12
	/** Haversine distance **/
	Haversine = 13
	/** Bray-Curtis distance **/
	BrayCurtis = 14
	/** Jensen-Shannon distance**/
	JensenShannon = 15
	/** Hamming distance **/
	HammingUnexpanded = 16
	/** KLDivergence **/
	KLDivergence = 17
	/** RusselRao **/
	RusselRaoExpanded = 18
	/** Dice-Sorensen distance **/
	DiceExpanded = 19
	/** Precomputed (special value) **/
	Precomputed = 100
)
