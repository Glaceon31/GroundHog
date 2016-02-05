import theano
import theano.tensor as TT
import numpy

x = TT.ftensor3('x')
#y = TT.ftensor3('y')
weight = theano.shared(numpy.asarray([[1.,2.,3.],[0.,1.,2.]], dtype=theano.config.floatX), name='w')
weight = weight.dimshuffle(0,1,'x')

k = weight*x
z = TT.sum(weight*x,axis=1)

func = theano.function(inputs=[x], outputs=[z,x.shape,weight.shape,k,weight, z.shape])

#print func([[1.,5.]])
#print func(numpy.asarray([[[1.,2.],[2.,3.],[0.,1.5]],[[-1.,2.],[2.,3.],[0.,1.5]]], dtype=theano.config.floatX))
#theano.printing.pydotprint(y, compact=True,outfile='sgraph.png', var_with_name_simple = True)

wb = TT.fvector()
k = TT.fvector('k')
M = TT.fmatrix('m')
beta = TT.fvector()
g = TT.fvector()
add = TT.fvector()
erase = TT.fvector()
location = TT.fvector()
simw = TT.fmatrix()
old = TT.fmatrix()
oldb = TT.fvector()

k_unit = k / ( TT.sqrt(TT.sum(k**2)) + 1e-5 )
k_unit = k_unit.dimshuffle(('x',0)) #T.patternbroadcast(k_unit.reshape((1,k_unit.shape[0])),(True,False))
k_unit.name = "k_unit"
M_lengths = TT.sqrt(TT.sum(M**2,axis=1)).dimshuffle((0,'x'))
M_unit = M / ( M_lengths + 1e-5 )
    
M_unit.name = "M_unit"
sim = TT.sum(k_unit * M_unit,axis=1)
weight_c = TT.nnet.softmax(beta.reshape((1,))*sim)
g = g.reshape((1,))
weight_g = g*weight_c+(1-g)*wb
weight_cross = TT.dot(weight_c, simw)+TT.dot(wb, old)+oldb
weight_cross = TT.nnet.softmax(weight_cross)
weight = weight_g
weight_dim = weight.reshape((weight.shape[1],)).dimshuffle((0, 'x'))
erase_dim = erase.reshape((erase.shape[0],)).dimshuffle(('x', 0))
add_dim = add.reshape((add.shape[0],)).dimshuffle(('x', 0))
memory_erase = M*(1-(weight_dim*erase_dim))
memory = memory_erase+(weight_dim*add_dim)

oldstate = TT.fvector()
w_state = TT.fmatrix()
w_memory = TT.fmatrix()
w_outer = TT.fmatrix()

m_vec = TT.dot(M, w_memory)
s_vec = TT.dot(oldstate,w_state)
inner = m_vec+s_vec
innert = TT.tanh(inner)
ot = TT.dot(innert, w_outer)
otexp = TT.exp(ot).reshape((ot.shape[0],))
normalizer = otexp.sum(axis=0)
result = otexp/normalizer

simfunc = theano.function(inputs=[wb,k,M,beta,g,simw,old,oldb,erase,add], outputs=[sim,weight_c,weight_cross,weight,memory_erase,memory])
attfunc = theano.function(inputs=[M,oldstate, w_state,w_memory,w_outer], outputs=[m_vec,s_vec,inner,innert,ot,otexp,result])
wbt = numpy.asarray([0.3,.7], dtype=theano.config.floatX)
kt = numpy.asarray([1.,.5,.2], dtype=theano.config.floatX)
Mt = numpy.asarray([[2,1,.4],[.1,.4,.1]], dtype=theano.config.floatX)
betat = numpy.asarray([1.], dtype=theano.config.floatX)
gt = numpy.asarray([0.], dtype=theano.config.floatX)
simt = numpy.asarray([[0.5,.5],[0,1]], dtype=theano.config.floatX)
oldbt = numpy.asarray([0.6,.8], dtype=theano.config.floatX)
oldt = numpy.asarray([[0.5,0.],[0,1]], dtype=theano.config.floatX)
eraset = numpy.asarray([.5,.5,.2], dtype=theano.config.floatX)
addt = numpy.asarray([1,1,.2], dtype=theano.config.floatX)
#print simfunc(wbt,kt, Mt,betat,gt,simt,oldt, oldbt,eraset,addt)
ost = numpy.asarray([0.6,.8], dtype=theano.config.floatX)
wst = numpy.asarray([[0.1,.2,.3],[.1,.4,.6]], dtype=theano.config.floatX)
wmt = numpy.asarray([[0.3,.2,.1],[.1,.4,.6],[.8,.1,.4]], dtype=theano.config.floatX)
outert = numpy.asarray([[0.7],[0.2],[0.6]], dtype=theano.config.floatX)
print attfunc(Mt,ost,wst,wmt,outert)

wb = TT.fmatrix()
k = TT.fmatrix('k')
M = TT.ftensor3('m')
beta = TT.fmatrix()
g = TT.fmatrix()
add = TT.fmatrix()
erase = TT.fmatrix()
simw = TT.fmatrix()
old = TT.fmatrix()
oldb = TT.fvector()

k_lengths = TT.sqrt(TT.sum(k**2,axis=1)).dimshuffle((0,'x'))
k_unit = k / ( k_lengths + 1e-5 )
k_unit = k_unit.dimshuffle((0,'x',1)) #T.patternbroadcast(k_unit.reshape((1,k_unit.shape[0])),(True,False))
k_unit.name = "k_unit"
M_lengths = TT.sqrt(TT.sum(M**2,axis=2)).dimshuffle((0,1,'x'))
M_unit = M / ( M_lengths + 1e-5 )
    
M_unit.name = "M_unit"

sim = TT.sum(k_unit * M_unit,axis=2)
bt = beta.reshape((beta.shape[0],)).dimshuffle(0,'x')*sim
weight_c = TT.nnet.softmax(bt)
g = g.reshape((g.shape[0],)).dimshuffle(0,'x')
weight_g = g*weight_c+(1-g)*wb
weight = weight_g
weight_cross = TT.dot(weight_c, simw)+TT.dot(wb, old)+oldb
weight_cross = TT.nnet.softmax(weight_cross)
M_erase = M*(1-weight.dimshuffle(0,1,'x')*erase.dimshuffle(0,'x',1))
memory = M_erase+weight.dimshuffle(0,1,'x')*add.dimshuffle(0,'x',1)
result = weight.dimshuffle(0,1,'x')*add.dimshuffle(0,'x',1)
simfunc = theano.function(inputs=[wb,k,M,beta,g,simw,old,oldb,erase,add], outputs=[sim,bt,weight_c,weight_cross,weight_g,M_erase,memory])
wbt = numpy.asarray([[0.3,.7],[0.,1.]], dtype=theano.config.floatX)
kt = numpy.asarray([[1.,.5,.2],[.3,.5,.7]], dtype=theano.config.floatX)
Mt = numpy.asarray([[[2,1,.4],[.1,.4,.1]],[[1,1.6,2.1],[.1,.1,-.1]]], dtype=theano.config.floatX)
betat = numpy.asarray([[1.],[1.3]], dtype=theano.config.floatX)
gt = numpy.asarray([[0,],[1.]], dtype=theano.config.floatX)
simt = numpy.asarray([[0.5,.5],[0,1]], dtype=theano.config.floatX)
oldbt = numpy.asarray([0.6,.8], dtype=theano.config.floatX)
oldt = numpy.asarray([[0.5,0.],[0,1]], dtype=theano.config.floatX)
eraset = numpy.asarray([[.5,.5,.2],[.3,.5,.7]], dtype=theano.config.floatX)
addt = numpy.asarray([[1,1,.2],[3,.5,3]], dtype=theano.config.floatX)
print simfunc(wbt,kt,Mt,betat,gt,simt,oldt, oldbt,eraset,addt)



'''
a = TT.fvector('a')
aa = TT.fvector('aa')
b = TT.alloc(a, 6,*a.shape)
bb = b+aa
func2 = theano.function(inputs=[a,aa], outputs=[bb])
in1 = numpy.asarray([1.,2.,5.], dtype=theano.config.floatX)
in2 = numpy.asarray([1.,3.,5.], dtype=theano.config.floatX)
#print func2(in1, in2)

c = TT.fmatrix('c')
#d = c.shape
d = TT.alloc(c, 8,*c.shape)
func3 = theano.function(inputs=[c], outputs=[d])

#print func3(numpy.asarray([[1.,2.,5.],[1.,2.,1.]], dtype=theano.config.floatX))

batch_in = numpy.asarray([[1.,2.,5.],[1.,2.,1.]], dtype=theano.config.floatX)
weight_in = numpy.asarray([[1.],[0.6],[2.]], dtype=theano.config.floatX)
bias_in = numpy.asarray([2], dtype=theano.config.floatX)

bat = TT.fmatrix('bat')
weight = TT.fmatrix('weight')
bias = TT.vector('bias')
out = TT.dot(bat,weight)+bias
out = out.reshape((out.shape[0],))
func4 = theano.function(inputs=[bat,weight, bias],outputs=[out])
print func4(batch_in,weight_in,bias_in)
'''
