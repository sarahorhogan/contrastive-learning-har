??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ۉ

z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:	 *
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
: *
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
: @*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:@*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:@`*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:`*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	`?*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	?2*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:2*
dtype0

NoOpNoOp
?,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?+
value?+B?+ B?+
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
R
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
R
1	variables
2trainable_variables
3regularization_losses
4	keras_api
h

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
R
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
h

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
V
0
1
2
3
'4
(5
56
67
?8
@9
I10
J11
V
0
1
2
3
'4
(5
56
67
?8
@9
I10
J11
 
?
Olayer_metrics
Pmetrics
Qlayer_regularization_losses

Rlayers
Snon_trainable_variables
	variables
trainable_variables
regularization_losses
 
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Tlayer_metrics
Umetrics
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
	variables
trainable_variables
regularization_losses
 
 
 
?
Ylayer_metrics
Zmetrics
[layer_regularization_losses

\layers
]non_trainable_variables
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
^layer_metrics
_metrics
`layer_regularization_losses

alayers
bnon_trainable_variables
	variables
 trainable_variables
!regularization_losses
 
 
 
?
clayer_metrics
dmetrics
elayer_regularization_losses

flayers
gnon_trainable_variables
#	variables
$trainable_variables
%regularization_losses
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
hlayer_metrics
imetrics
jlayer_regularization_losses

klayers
lnon_trainable_variables
)	variables
*trainable_variables
+regularization_losses
 
 
 
?
mlayer_metrics
nmetrics
olayer_regularization_losses

players
qnon_trainable_variables
-	variables
.trainable_variables
/regularization_losses
 
 
 
?
rlayer_metrics
smetrics
tlayer_regularization_losses

ulayers
vnon_trainable_variables
1	variables
2trainable_variables
3regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61

50
61
 
?
wlayer_metrics
xmetrics
ylayer_regularization_losses

zlayers
{non_trainable_variables
7	variables
8trainable_variables
9regularization_losses
 
 
 
?
|layer_metrics
}metrics
~layer_regularization_losses

layers
?non_trainable_variables
;	variables
<trainable_variables
=regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
?
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
A	variables
Btrainable_variables
Cregularization_losses
 
 
 
?
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
E	variables
Ftrainable_variables
Gregularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
?
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
K	variables
Ltrainable_variables
Mregularization_losses
 
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_inputPlaceholder*,
_output_shapes
:??????????	*
dtype0*!
shape:??????????	
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_8096
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*&
f!R
__inference__traced_save_8637
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_restore_8685??	
?
k
__inference_loss_fn_0_8548<
8conv1d_kernel_regularizer_square_readvariableop_resource
identity??
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv1d_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:	 *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOp?
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:	 2"
 conv1d/kernel/Regularizer/Square?
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Const?
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/Sum?
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82!
conv1d/kernel/Regularizer/mul/x?
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul?
conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv1d/kernel/Regularizer/add/x?
conv1d/kernel/Regularizer/addAddV2(conv1d/kernel/Regularizer/add/x:output:0!conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/addd
IdentityIdentity!conv1d/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_8408

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????Z@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????Z@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????Z@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Z@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????Z@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????Z@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????Z@:S O
+
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_7577

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????i 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????i 2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:?????????i :S O
+
_output_shapes
:?????????i 
 
_user_specified_nameinputs
?
D
(__inference_dropout_1_layer_call_fn_8423

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????Z@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_76122
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????Z@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????Z@:S O
+
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_7572

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????i 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????i *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????i 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????i 2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????i 2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????i 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????i :S O
+
_output_shapes
:?????????i 
 
_user_specified_nameinputs
?
?
A__inference_dense_2_layer_call_and_return_conditional_losses_8526

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
m
__inference_loss_fn_1_8561>
:conv1d_1_kernel_regularizer_square_readvariableop_resource
identity??
1conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv1d_1_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: @*
dtype023
1conv1d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_1/kernel/Regularizer/SquareSquare9conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2$
"conv1d_1/kernel/Regularizer/Square?
!conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_1/kernel/Regularizer/Const?
conv1d_1/kernel/Regularizer/SumSum&conv1d_1/kernel/Regularizer/Square:y:0*conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/Sum?
!conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_1/kernel/Regularizer/mul/x?
conv1d_1/kernel/Regularizer/mulMul*conv1d_1/kernel/Regularizer/mul/x:output:0(conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/mul?
!conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_1/kernel/Regularizer/add/x?
conv1d_1/kernel/Regularizer/addAddV2*conv1d_1/kernel/Regularizer/add/x:output:0#conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/addf
IdentityIdentity#conv1d_1/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?,
?
__inference__traced_save_8637
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_fc85003084844617bfb0c8f29c622b3f/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapesx
v: :	 : : @:@:@`:`:	`?:?:
??:?:	?2:2: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:	 : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@:($
"
_output_shapes
:@`: 

_output_shapes
:`:%!

_output_shapes
:	`?:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?2: 

_output_shapes
:2:

_output_shapes
: 
?
?
A__inference_dense_2_layer_call_and_return_conditional_losses_7749

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_7612

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????Z@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????Z@2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:?????????Z@:S O
+
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?9
?
 __inference__traced_restore_8685
file_prefix"
assignvariableop_conv1d_kernel"
assignvariableop_1_conv1d_bias&
"assignvariableop_2_conv1d_1_kernel$
 assignvariableop_3_conv1d_1_bias&
"assignvariableop_4_conv1d_2_kernel$
 assignvariableop_5_conv1d_2_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias%
!assignvariableop_8_dense_1_kernel#
assignvariableop_9_dense_1_bias&
"assignvariableop_10_dense_2_kernel$
 assignvariableop_11_dense_2_bias
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_2_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
z
%__inference_conv1d_layer_call_fn_7468

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_74582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????	::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
{
&__inference_dense_1_layer_call_fn_8506

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_77102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
{
&__inference_dense_2_layer_call_fn_8535

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_77492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_8206

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity?~
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????	2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	 *
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	 2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????i *
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????i *
squeeze_dims
2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????i 2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????i 2
conv1d/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/dropout/Const?
dropout/dropout/MulMulconv1d/Relu:activations:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????i 2
dropout/dropout/Mulw
dropout/dropout/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????i *
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????i 2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????i 2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????i 2
dropout/dropout/Mul_1?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsdropout/dropout/Mul_1:z:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????i 2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????Z@*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????Z@*
squeeze_dims
2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????Z@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????Z@2
conv1d_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulconv1d_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????Z@2
dropout_1/dropout/Mul}
dropout_1/dropout/ShapeShapeconv1d_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????Z@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????Z@2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Z@2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????Z@2
dropout_1/dropout/Mul_1?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsdropout_1/dropout/Mul_1:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????Z@2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????S`*
paddingVALID*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????S`*
squeeze_dims
2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????S`2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????S`2
conv1d_2/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulconv1d_2/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:?????????S`2
dropout_2/dropout/Mul}
dropout_2/dropout/ShapeShapeconv1d_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????S`*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????S`2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????S`2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????S`2
dropout_2/dropout/Mul_1?
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*global_max_pooling1d/Max/reduction_indices?
global_max_pooling1d/MaxMaxdropout_2/dropout/Mul_1:z:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????`2
global_max_pooling1d/Max?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	`?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul!global_max_pooling1d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddu
activation/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
activation/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulactivation/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAdd{
activation_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
activation_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulactivation_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_2/BiasAdd?
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	 *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOp?
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:	 2"
 conv1d/kernel/Regularizer/Square?
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Const?
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/Sum?
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82!
conv1d/kernel/Regularizer/mul/x?
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul?
conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv1d/kernel/Regularizer/add/x?
conv1d/kernel/Regularizer/addAddV2(conv1d/kernel/Regularizer/add/x:output:0!conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/add?
1conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype023
1conv1d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_1/kernel/Regularizer/SquareSquare9conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2$
"conv1d_1/kernel/Regularizer/Square?
!conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_1/kernel/Regularizer/Const?
conv1d_1/kernel/Regularizer/SumSum&conv1d_1/kernel/Regularizer/Square:y:0*conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/Sum?
!conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_1/kernel/Regularizer/mul/x?
conv1d_1/kernel/Regularizer/mulMul*conv1d_1/kernel/Regularizer/mul/x:output:0(conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/mul?
!conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_1/kernel/Regularizer/add/x?
conv1d_1/kernel/Regularizer/addAddV2*conv1d_1/kernel/Regularizer/add/x:output:0#conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/add?
1conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype023
1conv1d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_2/kernel/Regularizer/SquareSquare9conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2$
"conv1d_2/kernel/Regularizer/Square?
!conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_2/kernel/Regularizer/Const?
conv1d_2/kernel/Regularizer/SumSum&conv1d_2/kernel/Regularizer/Square:y:0*conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/Sum?
!conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_2/kernel/Regularizer/mul/x?
conv1d_2/kernel/Regularizer/mulMul*conv1d_2/kernel/Regularizer/mul/x:output:0(conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/mul?
!conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_2/kernel/Regularizer/add/x?
conv1d_2/kernel/Regularizer/addAddV2*conv1d_2/kernel/Regularizer/add/x:output:0#conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/addl
IdentityIdentitydense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	:::::::::::::T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
D
(__inference_dropout_2_layer_call_fn_8458

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????S`* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_76472
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????S`2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????S`:S O
+
_output_shapes
:?????????S`
 
_user_specified_nameinputs
?Q
?
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_8014

inputs
conv1d_7953
conv1d_7955
conv1d_1_7959
conv1d_1_7961
conv1d_2_7965
conv1d_2_7967

dense_7972

dense_7974
dense_1_7978
dense_1_7980
dense_2_7984
dense_2_7986
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_7953conv1d_7955*
Tin
2*
Tout
2*+
_output_shapes
:?????????i *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_74582 
conv1d/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????i * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_75772
dropout/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_1_7959conv1d_1_7961*
Tin
2*
Tout
2*+
_output_shapes
:?????????Z@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_74932"
 conv1d_1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????Z@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_76122
dropout_1/PartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_2_7965conv1d_2_7967*
Tin
2*
Tout
2*+
_output_shapes
:?????????S`*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_75282"
 conv1d_2/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????S`* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_76472
dropout_2/PartitionedCall?
$global_max_pooling1d/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????`* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_75452&
$global_max_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_7972
dense_7974*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_76712
dense/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_76922
activation/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_7978dense_1_7980*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_77102!
dense_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_77312
activation_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0dense_2_7984dense_2_7986*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_77492!
dense_2/StatefulPartitionedCall?
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_7953*"
_output_shapes
:	 *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOp?
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:	 2"
 conv1d/kernel/Regularizer/Square?
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Const?
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/Sum?
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82!
conv1d/kernel/Regularizer/mul/x?
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul?
conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv1d/kernel/Regularizer/add/x?
conv1d/kernel/Regularizer/addAddV2(conv1d/kernel/Regularizer/add/x:output:0!conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/add?
1conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_1_7959*"
_output_shapes
: @*
dtype023
1conv1d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_1/kernel/Regularizer/SquareSquare9conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2$
"conv1d_1/kernel/Regularizer/Square?
!conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_1/kernel/Regularizer/Const?
conv1d_1/kernel/Regularizer/SumSum&conv1d_1/kernel/Regularizer/Square:y:0*conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/Sum?
!conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_1/kernel/Regularizer/mul/x?
conv1d_1/kernel/Regularizer/mulMul*conv1d_1/kernel/Regularizer/mul/x:output:0(conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/mul?
!conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_1/kernel/Regularizer/add/x?
conv1d_1/kernel/Regularizer/addAddV2*conv1d_1/kernel/Regularizer/add/x:output:0#conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/add?
1conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_2_7965*"
_output_shapes
:@`*
dtype023
1conv1d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_2/kernel/Regularizer/SquareSquare9conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2$
"conv1d_2/kernel/Regularizer/Square?
!conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_2/kernel/Regularizer/Const?
conv1d_2/kernel/Regularizer/SumSum&conv1d_2/kernel/Regularizer/Square:y:0*conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/Sum?
!conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_2/kernel/Regularizer/mul/x?
conv1d_2/kernel/Regularizer/mulMul*conv1d_2/kernel/Regularizer/mul/x:output:0(conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/mul?
!conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_2/kernel/Regularizer/add/x?
conv1d_2/kernel/Regularizer/addAddV2*conv1d_2/kernel/Regularizer/add/x:output:0#conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/add?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_8443

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????S`2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????S`*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????S`2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????S`2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????S`2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????S`2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????S`:S O
+
_output_shapes
:?????????S`
 
_user_specified_nameinputs
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_7607

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????Z@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????Z@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????Z@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Z@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????Z@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????Z@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????Z@:S O
+
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?V
?
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_7790	
input
conv1d_7555
conv1d_7557
conv1d_1_7590
conv1d_1_7592
conv1d_2_7625
conv1d_2_7627

dense_7682

dense_7684
dense_1_7721
dense_1_7723
dense_2_7760
dense_2_7762
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputconv1d_7555conv1d_7557*
Tin
2*
Tout
2*+
_output_shapes
:?????????i *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_74582 
conv1d/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????i * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_75722!
dropout/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_1_7590conv1d_1_7592*
Tin
2*
Tout
2*+
_output_shapes
:?????????Z@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_74932"
 conv1d_1/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*+
_output_shapes
:?????????Z@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_76072#
!dropout_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_2_7625conv1d_2_7627*
Tin
2*
Tout
2*+
_output_shapes
:?????????S`*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_75282"
 conv1d_2/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*+
_output_shapes
:?????????S`* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_76422#
!dropout_2/StatefulPartitionedCall?
$global_max_pooling1d/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????`* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_75452&
$global_max_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_7682
dense_7684*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_76712
dense/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_76922
activation/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_7721dense_1_7723*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_77102!
dense_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_77312
activation_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0dense_2_7760dense_2_7762*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_77492!
dense_2/StatefulPartitionedCall?
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_7555*"
_output_shapes
:	 *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOp?
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:	 2"
 conv1d/kernel/Regularizer/Square?
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Const?
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/Sum?
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82!
conv1d/kernel/Regularizer/mul/x?
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul?
conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv1d/kernel/Regularizer/add/x?
conv1d/kernel/Regularizer/addAddV2(conv1d/kernel/Regularizer/add/x:output:0!conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/add?
1conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_1_7590*"
_output_shapes
: @*
dtype023
1conv1d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_1/kernel/Regularizer/SquareSquare9conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2$
"conv1d_1/kernel/Regularizer/Square?
!conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_1/kernel/Regularizer/Const?
conv1d_1/kernel/Regularizer/SumSum&conv1d_1/kernel/Regularizer/Square:y:0*conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/Sum?
!conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_1/kernel/Regularizer/mul/x?
conv1d_1/kernel/Regularizer/mulMul*conv1d_1/kernel/Regularizer/mul/x:output:0(conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/mul?
!conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_1/kernel/Regularizer/add/x?
conv1d_1/kernel/Regularizer/addAddV2*conv1d_1/kernel/Regularizer/add/x:output:0#conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/add?
1conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_2_7625*"
_output_shapes
:@`*
dtype023
1conv1d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_2/kernel/Regularizer/SquareSquare9conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2$
"conv1d_2/kernel/Regularizer/Square?
!conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_2/kernel/Regularizer/Const?
conv1d_2/kernel/Regularizer/SumSum&conv1d_2/kernel/Regularizer/Square:y:0*conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/Sum?
!conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_2/kernel/Regularizer/mul/x?
conv1d_2/kernel/Regularizer/mulMul*conv1d_2/kernel/Regularizer/mul/x:output:0(conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/mul?
!conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_2/kernel/Regularizer/add/x?
conv1d_2/kernel/Regularizer/addAddV2*conv1d_2/kernel/Regularizer/add/x:output:0#conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/add?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:S O
,
_output_shapes
:??????????	

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_8373

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????i 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????i *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????i 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????i 2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????i 2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????i 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????i :S O
+
_output_shapes
:?????????i 
 
_user_specified_nameinputs
?
?
0__inference_base_model_simclr_layer_call_fn_7948	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_79212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:??????????	

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_dense_1_layer_call_and_return_conditional_losses_8497

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
m
__inference_loss_fn_2_8574>
:conv1d_2_kernel_regularizer_square_readvariableop_resource
identity??
1conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv1d_2_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@`*
dtype023
1conv1d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_2/kernel/Regularizer/SquareSquare9conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2$
"conv1d_2/kernel/Regularizer/Square?
!conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_2/kernel/Regularizer/Const?
conv1d_2/kernel/Regularizer/SumSum&conv1d_2/kernel/Regularizer/Square:y:0*conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/Sum?
!conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_2/kernel/Regularizer/mul/x?
conv1d_2/kernel/Regularizer/mulMul*conv1d_2/kernel/Regularizer/mul/x:output:0(conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/mul?
!conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_2/kernel/Regularizer/add/x?
conv1d_2/kernel/Regularizer/addAddV2*conv1d_2/kernel/Regularizer/add/x:output:0#conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/addf
IdentityIdentity#conv1d_2/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
a
(__inference_dropout_2_layer_call_fn_8453

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????S`* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_76422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????S`2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????S`22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????S`
 
_user_specified_nameinputs
?
?
B__inference_conv1d_2_layer_call_and_return_conditional_losses_7528

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????`*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :??????????????????`*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????`2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????`2
Relu?
1conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype023
1conv1d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_2/kernel/Regularizer/SquareSquare9conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2$
"conv1d_2/kernel/Regularizer/Square?
!conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_2/kernel/Regularizer/Const?
conv1d_2/kernel/Regularizer/SumSum&conv1d_2/kernel/Regularizer/Square:y:0*conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/Sum?
!conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_2/kernel/Regularizer/mul/x?
conv1d_2/kernel/Regularizer/mulMul*conv1d_2/kernel/Regularizer/mul/x:output:0(conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/mul?
!conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_2/kernel/Regularizer/add/x?
conv1d_2/kernel/Regularizer/addAddV2*conv1d_2/kernel/Regularizer/add/x:output:0#conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????`2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????@:::\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
`
D__inference_activation_layer_call_and_return_conditional_losses_8482

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_8096	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_74332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:??????????	

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
a
(__inference_dropout_1_layer_call_fn_8418

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????Z@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_76072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????Z@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????Z@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?
G
+__inference_activation_1_layer_call_fn_8516

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_77312
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_activation_1_layer_call_and_return_conditional_losses_7731

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_activation_layer_call_and_return_conditional_losses_7692

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
y
$__inference_dense_layer_call_fn_8477

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_76712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????`::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_7647

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????S`2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????S`2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:?????????S`:S O
+
_output_shapes
:?????????S`
 
_user_specified_nameinputs
?
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_7642

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????S`2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????S`*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????S`2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????S`2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????S`2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????S`2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????S`:S O
+
_output_shapes
:?????????S`
 
_user_specified_nameinputs
?
j
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_7545

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
_
&__inference_dropout_layer_call_fn_8383

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????i * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_75722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????i 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????i 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????i 
 
_user_specified_nameinputs
?
?
0__inference_base_model_simclr_layer_call_fn_8324

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_79212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_8378

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????i 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????i 2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:?????????i :S O
+
_output_shapes
:?????????i 
 
_user_specified_nameinputs
?
?
?__inference_dense_layer_call_and_return_conditional_losses_7671

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	`?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????`:::O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?V
?
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_7921

inputs
conv1d_7860
conv1d_7862
conv1d_1_7866
conv1d_1_7868
conv1d_2_7872
conv1d_2_7874

dense_7879

dense_7881
dense_1_7885
dense_1_7887
dense_2_7891
dense_2_7893
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_7860conv1d_7862*
Tin
2*
Tout
2*+
_output_shapes
:?????????i *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_74582 
conv1d/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????i * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_75722!
dropout/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_1_7866conv1d_1_7868*
Tin
2*
Tout
2*+
_output_shapes
:?????????Z@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_74932"
 conv1d_1/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*+
_output_shapes
:?????????Z@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_76072#
!dropout_1/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_2_7872conv1d_2_7874*
Tin
2*
Tout
2*+
_output_shapes
:?????????S`*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_75282"
 conv1d_2/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*+
_output_shapes
:?????????S`* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_76422#
!dropout_2/StatefulPartitionedCall?
$global_max_pooling1d/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????`* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_75452&
$global_max_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_7879
dense_7881*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_76712
dense/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_76922
activation/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_7885dense_1_7887*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_77102!
dense_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_77312
activation_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0dense_2_7891dense_2_7893*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_77492!
dense_2/StatefulPartitionedCall?
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_7860*"
_output_shapes
:	 *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOp?
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:	 2"
 conv1d/kernel/Regularizer/Square?
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Const?
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/Sum?
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82!
conv1d/kernel/Regularizer/mul/x?
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul?
conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv1d/kernel/Regularizer/add/x?
conv1d/kernel/Regularizer/addAddV2(conv1d/kernel/Regularizer/add/x:output:0!conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/add?
1conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_1_7866*"
_output_shapes
: @*
dtype023
1conv1d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_1/kernel/Regularizer/SquareSquare9conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2$
"conv1d_1/kernel/Regularizer/Square?
!conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_1/kernel/Regularizer/Const?
conv1d_1/kernel/Regularizer/SumSum&conv1d_1/kernel/Regularizer/Square:y:0*conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/Sum?
!conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_1/kernel/Regularizer/mul/x?
conv1d_1/kernel/Regularizer/mulMul*conv1d_1/kernel/Regularizer/mul/x:output:0(conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/mul?
!conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_1/kernel/Regularizer/add/x?
conv1d_1/kernel/Regularizer/addAddV2*conv1d_1/kernel/Regularizer/add/x:output:0#conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/add?
1conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_2_7872*"
_output_shapes
:@`*
dtype023
1conv1d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_2/kernel/Regularizer/SquareSquare9conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2$
"conv1d_2/kernel/Regularizer/Square?
!conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_2/kernel/Regularizer/Const?
conv1d_2/kernel/Regularizer/SumSum&conv1d_2/kernel/Regularizer/Square:y:0*conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/Sum?
!conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_2/kernel/Regularizer/mul/x?
conv1d_2/kernel/Regularizer/mulMul*conv1d_2/kernel/Regularizer/mul/x:output:0(conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/mul?
!conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_2/kernel/Regularizer/add/x?
conv1d_2/kernel/Regularizer/addAddV2*conv1d_2/kernel/Regularizer/add/x:output:0#conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/add?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_conv1d_2_layer_call_fn_7538

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????????????`*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_75282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????`2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
0__inference_base_model_simclr_layer_call_fn_8041	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_80142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:??????????	

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_conv1d_1_layer_call_and_return_conditional_losses_7493

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????@2
Relu?
1conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype023
1conv1d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_1/kernel/Regularizer/SquareSquare9conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2$
"conv1d_1/kernel/Regularizer/Square?
!conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_1/kernel/Regularizer/Const?
conv1d_1/kernel/Regularizer/SumSum&conv1d_1/kernel/Regularizer/Square:y:0*conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/Sum?
!conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_1/kernel/Regularizer/mul/x?
conv1d_1/kernel/Regularizer/mulMul*conv1d_1/kernel/Regularizer/mul/x:output:0(conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/mul?
!conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_1/kernel/Regularizer/add/x?
conv1d_1/kernel/Regularizer/addAddV2*conv1d_1/kernel/Regularizer/add/x:output:0#conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????????????? :::\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
?__inference_dense_layer_call_and_return_conditional_losses_8468

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	`?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????`:::O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
@__inference_conv1d_layer_call_and_return_conditional_losses_7458

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????	2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	 *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :?????????????????? *
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
Relu?
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	 *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOp?
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:	 2"
 conv1d/kernel/Regularizer/Square?
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Const?
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/Sum?
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82!
conv1d/kernel/Regularizer/mul/x?
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul?
conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv1d/kernel/Regularizer/add/x?
conv1d/kernel/Regularizer/addAddV2(conv1d/kernel/Regularizer/add/x:output:0!conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/adds
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????	:::\ X
4
_output_shapes"
 :??????????????????	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?k
?
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_8295

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity?~
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????	2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	 *
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	 2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????i *
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????i *
squeeze_dims
2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????i 2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????i 2
conv1d/Relu?
dropout/IdentityIdentityconv1d/Relu:activations:0*
T0*+
_output_shapes
:?????????i 2
dropout/Identity?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsdropout/Identity:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????i 2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????Z@*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????Z@*
squeeze_dims
2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????Z@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????Z@2
conv1d_1/Relu?
dropout_1/IdentityIdentityconv1d_1/Relu:activations:0*
T0*+
_output_shapes
:?????????Z@2
dropout_1/Identity?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsdropout_1/Identity:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????Z@2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????S`*
paddingVALID*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????S`*
squeeze_dims
2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????S`2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????S`2
conv1d_2/Relu?
dropout_2/IdentityIdentityconv1d_2/Relu:activations:0*
T0*+
_output_shapes
:?????????S`2
dropout_2/Identity?
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*global_max_pooling1d/Max/reduction_indices?
global_max_pooling1d/MaxMaxdropout_2/Identity:output:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????`2
global_max_pooling1d/Max?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	`?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul!global_max_pooling1d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddu
activation/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
activation/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulactivation/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAdd{
activation_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
activation_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulactivation_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_2/BiasAdd?
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	 *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOp?
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:	 2"
 conv1d/kernel/Regularizer/Square?
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Const?
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/Sum?
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82!
conv1d/kernel/Regularizer/mul/x?
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul?
conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv1d/kernel/Regularizer/add/x?
conv1d/kernel/Regularizer/addAddV2(conv1d/kernel/Regularizer/add/x:output:0!conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/add?
1conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype023
1conv1d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_1/kernel/Regularizer/SquareSquare9conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2$
"conv1d_1/kernel/Regularizer/Square?
!conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_1/kernel/Regularizer/Const?
conv1d_1/kernel/Regularizer/SumSum&conv1d_1/kernel/Regularizer/Square:y:0*conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/Sum?
!conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_1/kernel/Regularizer/mul/x?
conv1d_1/kernel/Regularizer/mulMul*conv1d_1/kernel/Regularizer/mul/x:output:0(conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/mul?
!conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_1/kernel/Regularizer/add/x?
conv1d_1/kernel/Regularizer/addAddV2*conv1d_1/kernel/Regularizer/add/x:output:0#conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/add?
1conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype023
1conv1d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_2/kernel/Regularizer/SquareSquare9conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2$
"conv1d_2/kernel/Regularizer/Square?
!conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_2/kernel/Regularizer/Const?
conv1d_2/kernel/Regularizer/SumSum&conv1d_2/kernel/Regularizer/Square:y:0*conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/Sum?
!conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_2/kernel/Regularizer/mul/x?
conv1d_2/kernel/Regularizer/mulMul*conv1d_2/kernel/Regularizer/mul/x:output:0(conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/mul?
!conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_2/kernel/Regularizer/add/x?
conv1d_2/kernel/Regularizer/addAddV2*conv1d_2/kernel/Regularizer/add/x:output:0#conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/addl
IdentityIdentitydense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	:::::::::::::T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_conv1d_1_layer_call_fn_7503

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_74932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?Q
?
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_7854	
input
conv1d_7793
conv1d_7795
conv1d_1_7799
conv1d_1_7801
conv1d_2_7805
conv1d_2_7807

dense_7812

dense_7814
dense_1_7818
dense_1_7820
dense_2_7824
dense_2_7826
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputconv1d_7793conv1d_7795*
Tin
2*
Tout
2*+
_output_shapes
:?????????i *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_74582 
conv1d/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????i * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_75772
dropout/PartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_1_7799conv1d_1_7801*
Tin
2*
Tout
2*+
_output_shapes
:?????????Z@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv1d_1_layer_call_and_return_conditional_losses_74932"
 conv1d_1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????Z@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_76122
dropout_1/PartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_2_7805conv1d_2_7807*
Tin
2*
Tout
2*+
_output_shapes
:?????????S`*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv1d_2_layer_call_and_return_conditional_losses_75282"
 conv1d_2/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????S`* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_76472
dropout_2/PartitionedCall?
$global_max_pooling1d/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????`* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_75452&
$global_max_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_7812
dense_7814*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_76712
dense/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_76922
activation/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_7818dense_1_7820*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_77102!
dense_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_77312
activation_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0dense_2_7824dense_2_7826*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_77492!
dense_2/StatefulPartitionedCall?
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_7793*"
_output_shapes
:	 *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOp?
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:	 2"
 conv1d/kernel/Regularizer/Square?
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Const?
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/Sum?
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82!
conv1d/kernel/Regularizer/mul/x?
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul?
conv1d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv1d/kernel/Regularizer/add/x?
conv1d/kernel/Regularizer/addAddV2(conv1d/kernel/Regularizer/add/x:output:0!conv1d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/add?
1conv1d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_1_7799*"
_output_shapes
: @*
dtype023
1conv1d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_1/kernel/Regularizer/SquareSquare9conv1d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2$
"conv1d_1/kernel/Regularizer/Square?
!conv1d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_1/kernel/Regularizer/Const?
conv1d_1/kernel/Regularizer/SumSum&conv1d_1/kernel/Regularizer/Square:y:0*conv1d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/Sum?
!conv1d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_1/kernel/Regularizer/mul/x?
conv1d_1/kernel/Regularizer/mulMul*conv1d_1/kernel/Regularizer/mul/x:output:0(conv1d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/mul?
!conv1d_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_1/kernel/Regularizer/add/x?
conv1d_1/kernel/Regularizer/addAddV2*conv1d_1/kernel/Regularizer/add/x:output:0#conv1d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_1/kernel/Regularizer/add?
1conv1d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_2_7805*"
_output_shapes
:@`*
dtype023
1conv1d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv1d_2/kernel/Regularizer/SquareSquare9conv1d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2$
"conv1d_2/kernel/Regularizer/Square?
!conv1d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!conv1d_2/kernel/Regularizer/Const?
conv1d_2/kernel/Regularizer/SumSum&conv1d_2/kernel/Regularizer/Square:y:0*conv1d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/Sum?
!conv1d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82#
!conv1d_2/kernel/Regularizer/mul/x?
conv1d_2/kernel/Regularizer/mulMul*conv1d_2/kernel/Regularizer/mul/x:output:0(conv1d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/mul?
!conv1d_2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv1d_2/kernel/Regularizer/add/x?
conv1d_2/kernel/Regularizer/addAddV2*conv1d_2/kernel/Regularizer/add/x:output:0#conv1d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv1d_2/kernel/Regularizer/add?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:S O
,
_output_shapes
:??????????	

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
0__inference_base_model_simclr_layer_call_fn_8353

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:?????????2*.
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_80142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
F__inference_activation_1_layer_call_and_return_conditional_losses_8511

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_8448

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????S`2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????S`2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:?????????S`:S O
+
_output_shapes
:?????????S`
 
_user_specified_nameinputs
?
O
3__inference_global_max_pooling1d_layer_call_fn_7551

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_75452
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_8388

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????i * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_75772
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????i 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????i :S O
+
_output_shapes
:?????????i 
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_8413

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????Z@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????Z@2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:?????????Z@:S O
+
_output_shapes
:?????????Z@
 
_user_specified_nameinputs
?
E
)__inference_activation_layer_call_fn_8487

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_76922
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_dense_1_layer_call_and_return_conditional_losses_7710

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?b
?
__inference__wrapped_model_7433	
inputH
Dbase_model_simclr_conv1d_conv1d_expanddims_1_readvariableop_resource<
8base_model_simclr_conv1d_biasadd_readvariableop_resourceJ
Fbase_model_simclr_conv1d_1_conv1d_expanddims_1_readvariableop_resource>
:base_model_simclr_conv1d_1_biasadd_readvariableop_resourceJ
Fbase_model_simclr_conv1d_2_conv1d_expanddims_1_readvariableop_resource>
:base_model_simclr_conv1d_2_biasadd_readvariableop_resource:
6base_model_simclr_dense_matmul_readvariableop_resource;
7base_model_simclr_dense_biasadd_readvariableop_resource<
8base_model_simclr_dense_1_matmul_readvariableop_resource=
9base_model_simclr_dense_1_biasadd_readvariableop_resource<
8base_model_simclr_dense_2_matmul_readvariableop_resource=
9base_model_simclr_dense_2_biasadd_readvariableop_resource
identity??
.base_model_simclr/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :20
.base_model_simclr/conv1d/conv1d/ExpandDims/dim?
*base_model_simclr/conv1d/conv1d/ExpandDims
ExpandDimsinput7base_model_simclr/conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????	2,
*base_model_simclr/conv1d/conv1d/ExpandDims?
;base_model_simclr/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpDbase_model_simclr_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	 *
dtype02=
;base_model_simclr/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
0base_model_simclr/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 22
0base_model_simclr/conv1d/conv1d/ExpandDims_1/dim?
,base_model_simclr/conv1d/conv1d/ExpandDims_1
ExpandDimsCbase_model_simclr/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:09base_model_simclr/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	 2.
,base_model_simclr/conv1d/conv1d/ExpandDims_1?
base_model_simclr/conv1d/conv1dConv2D3base_model_simclr/conv1d/conv1d/ExpandDims:output:05base_model_simclr/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????i *
paddingVALID*
strides
2!
base_model_simclr/conv1d/conv1d?
'base_model_simclr/conv1d/conv1d/SqueezeSqueeze(base_model_simclr/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????i *
squeeze_dims
2)
'base_model_simclr/conv1d/conv1d/Squeeze?
/base_model_simclr/conv1d/BiasAdd/ReadVariableOpReadVariableOp8base_model_simclr_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/base_model_simclr/conv1d/BiasAdd/ReadVariableOp?
 base_model_simclr/conv1d/BiasAddBiasAdd0base_model_simclr/conv1d/conv1d/Squeeze:output:07base_model_simclr/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????i 2"
 base_model_simclr/conv1d/BiasAdd?
base_model_simclr/conv1d/ReluRelu)base_model_simclr/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????i 2
base_model_simclr/conv1d/Relu?
"base_model_simclr/dropout/IdentityIdentity+base_model_simclr/conv1d/Relu:activations:0*
T0*+
_output_shapes
:?????????i 2$
"base_model_simclr/dropout/Identity?
0base_model_simclr/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0base_model_simclr/conv1d_1/conv1d/ExpandDims/dim?
,base_model_simclr/conv1d_1/conv1d/ExpandDims
ExpandDims+base_model_simclr/dropout/Identity:output:09base_model_simclr/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????i 2.
,base_model_simclr/conv1d_1/conv1d/ExpandDims?
=base_model_simclr/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFbase_model_simclr_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02?
=base_model_simclr/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
2base_model_simclr/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2base_model_simclr/conv1d_1/conv1d/ExpandDims_1/dim?
.base_model_simclr/conv1d_1/conv1d/ExpandDims_1
ExpandDimsEbase_model_simclr/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0;base_model_simclr/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @20
.base_model_simclr/conv1d_1/conv1d/ExpandDims_1?
!base_model_simclr/conv1d_1/conv1dConv2D5base_model_simclr/conv1d_1/conv1d/ExpandDims:output:07base_model_simclr/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????Z@*
paddingVALID*
strides
2#
!base_model_simclr/conv1d_1/conv1d?
)base_model_simclr/conv1d_1/conv1d/SqueezeSqueeze*base_model_simclr/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????Z@*
squeeze_dims
2+
)base_model_simclr/conv1d_1/conv1d/Squeeze?
1base_model_simclr/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp:base_model_simclr_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1base_model_simclr/conv1d_1/BiasAdd/ReadVariableOp?
"base_model_simclr/conv1d_1/BiasAddBiasAdd2base_model_simclr/conv1d_1/conv1d/Squeeze:output:09base_model_simclr/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????Z@2$
"base_model_simclr/conv1d_1/BiasAdd?
base_model_simclr/conv1d_1/ReluRelu+base_model_simclr/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????Z@2!
base_model_simclr/conv1d_1/Relu?
$base_model_simclr/dropout_1/IdentityIdentity-base_model_simclr/conv1d_1/Relu:activations:0*
T0*+
_output_shapes
:?????????Z@2&
$base_model_simclr/dropout_1/Identity?
0base_model_simclr/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0base_model_simclr/conv1d_2/conv1d/ExpandDims/dim?
,base_model_simclr/conv1d_2/conv1d/ExpandDims
ExpandDims-base_model_simclr/dropout_1/Identity:output:09base_model_simclr/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????Z@2.
,base_model_simclr/conv1d_2/conv1d/ExpandDims?
=base_model_simclr/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFbase_model_simclr_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02?
=base_model_simclr/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
2base_model_simclr/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2base_model_simclr/conv1d_2/conv1d/ExpandDims_1/dim?
.base_model_simclr/conv1d_2/conv1d/ExpandDims_1
ExpandDimsEbase_model_simclr/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0;base_model_simclr/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`20
.base_model_simclr/conv1d_2/conv1d/ExpandDims_1?
!base_model_simclr/conv1d_2/conv1dConv2D5base_model_simclr/conv1d_2/conv1d/ExpandDims:output:07base_model_simclr/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????S`*
paddingVALID*
strides
2#
!base_model_simclr/conv1d_2/conv1d?
)base_model_simclr/conv1d_2/conv1d/SqueezeSqueeze*base_model_simclr/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????S`*
squeeze_dims
2+
)base_model_simclr/conv1d_2/conv1d/Squeeze?
1base_model_simclr/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp:base_model_simclr_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype023
1base_model_simclr/conv1d_2/BiasAdd/ReadVariableOp?
"base_model_simclr/conv1d_2/BiasAddBiasAdd2base_model_simclr/conv1d_2/conv1d/Squeeze:output:09base_model_simclr/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????S`2$
"base_model_simclr/conv1d_2/BiasAdd?
base_model_simclr/conv1d_2/ReluRelu+base_model_simclr/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????S`2!
base_model_simclr/conv1d_2/Relu?
$base_model_simclr/dropout_2/IdentityIdentity-base_model_simclr/conv1d_2/Relu:activations:0*
T0*+
_output_shapes
:?????????S`2&
$base_model_simclr/dropout_2/Identity?
<base_model_simclr/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2>
<base_model_simclr/global_max_pooling1d/Max/reduction_indices?
*base_model_simclr/global_max_pooling1d/MaxMax-base_model_simclr/dropout_2/Identity:output:0Ebase_model_simclr/global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:?????????`2,
*base_model_simclr/global_max_pooling1d/Max?
-base_model_simclr/dense/MatMul/ReadVariableOpReadVariableOp6base_model_simclr_dense_matmul_readvariableop_resource*
_output_shapes
:	`?*
dtype02/
-base_model_simclr/dense/MatMul/ReadVariableOp?
base_model_simclr/dense/MatMulMatMul3base_model_simclr/global_max_pooling1d/Max:output:05base_model_simclr/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
base_model_simclr/dense/MatMul?
.base_model_simclr/dense/BiasAdd/ReadVariableOpReadVariableOp7base_model_simclr_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.base_model_simclr/dense/BiasAdd/ReadVariableOp?
base_model_simclr/dense/BiasAddBiasAdd(base_model_simclr/dense/MatMul:product:06base_model_simclr/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
base_model_simclr/dense/BiasAdd?
!base_model_simclr/activation/ReluRelu(base_model_simclr/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2#
!base_model_simclr/activation/Relu?
/base_model_simclr/dense_1/MatMul/ReadVariableOpReadVariableOp8base_model_simclr_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/base_model_simclr/dense_1/MatMul/ReadVariableOp?
 base_model_simclr/dense_1/MatMulMatMul/base_model_simclr/activation/Relu:activations:07base_model_simclr/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 base_model_simclr/dense_1/MatMul?
0base_model_simclr/dense_1/BiasAdd/ReadVariableOpReadVariableOp9base_model_simclr_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0base_model_simclr/dense_1/BiasAdd/ReadVariableOp?
!base_model_simclr/dense_1/BiasAddBiasAdd*base_model_simclr/dense_1/MatMul:product:08base_model_simclr/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!base_model_simclr/dense_1/BiasAdd?
#base_model_simclr/activation_1/ReluRelu*base_model_simclr/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2%
#base_model_simclr/activation_1/Relu?
/base_model_simclr/dense_2/MatMul/ReadVariableOpReadVariableOp8base_model_simclr_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype021
/base_model_simclr/dense_2/MatMul/ReadVariableOp?
 base_model_simclr/dense_2/MatMulMatMul1base_model_simclr/activation_1/Relu:activations:07base_model_simclr/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 base_model_simclr/dense_2/MatMul?
0base_model_simclr/dense_2/BiasAdd/ReadVariableOpReadVariableOp9base_model_simclr_dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype022
0base_model_simclr/dense_2/BiasAdd/ReadVariableOp?
!base_model_simclr/dense_2/BiasAddBiasAdd*base_model_simclr/dense_2/MatMul:product:08base_model_simclr/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!base_model_simclr/dense_2/BiasAdd~
IdentityIdentity*base_model_simclr/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:??????????	:::::::::::::S O
,
_output_shapes
:??????????	

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input3
serving_default_input:0??????????	;
dense_20
StatefulPartitionedCall:0?????????2tensorflow/serving/predict:??
?[
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?W
_tf_keras_model?W{"class_name": "Model", "name": "base_model_simclr", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "base_model_simclr", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "global_max_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling1d", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_max_pooling1d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 9]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "base_model_simclr", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "global_max_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling1d", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_max_pooling1d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 9]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?


kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 105, 32]}}
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?	

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 64]}}
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "global_max_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96]}}
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
v
0
1
2
3
'4
(5
56
67
?8
@9
I10
J11"
trackable_list_wrapper
v
0
1
2
3
'4
(5
56
67
?8
@9
I10
J11"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
Olayer_metrics
Pmetrics
Qlayer_regularization_losses

Rlayers
Snon_trainable_variables
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
#:!	 2conv1d/kernel
: 2conv1d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
Tlayer_metrics
Umetrics
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ylayer_metrics
Zmetrics
[layer_regularization_losses

\layers
]non_trainable_variables
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:# @2conv1d_1/kernel
:@2conv1d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
^layer_metrics
_metrics
`layer_regularization_losses

alayers
bnon_trainable_variables
	variables
 trainable_variables
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
clayer_metrics
dmetrics
elayer_regularization_losses

flayers
gnon_trainable_variables
#	variables
$trainable_variables
%regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#@`2conv1d_2/kernel
:`2conv1d_2/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
hlayer_metrics
imetrics
jlayer_regularization_losses

klayers
lnon_trainable_variables
)	variables
*trainable_variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mlayer_metrics
nmetrics
olayer_regularization_losses

players
qnon_trainable_variables
-	variables
.trainable_variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
rlayer_metrics
smetrics
tlayer_regularization_losses

ulayers
vnon_trainable_variables
1	variables
2trainable_variables
3regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	`?2dense/kernel
:?2
dense/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
wlayer_metrics
xmetrics
ylayer_regularization_losses

zlayers
{non_trainable_variables
7	variables
8trainable_variables
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|layer_metrics
}metrics
~layer_regularization_losses

layers
?non_trainable_variables
;	variables
<trainable_variables
=regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_1/kernel
:?2dense_1/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
A	variables
Btrainable_variables
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
E	variables
Ftrainable_variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?22dense_2/kernel
:22dense_2/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?non_trainable_variables
K	variables
Ltrainable_variables
Mregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
__inference__wrapped_model_7433?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *)?&
$?!
input??????????	
?2?
0__inference_base_model_simclr_layer_call_fn_8324
0__inference_base_model_simclr_layer_call_fn_8353
0__inference_base_model_simclr_layer_call_fn_8041
0__inference_base_model_simclr_layer_call_fn_7948?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_8295
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_8206
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_7790
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_7854?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_conv1d_layer_call_fn_7468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????	
?2?
@__inference_conv1d_layer_call_and_return_conditional_losses_7458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????	
?2?
&__inference_dropout_layer_call_fn_8388
&__inference_dropout_layer_call_fn_8383?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_8373
A__inference_dropout_layer_call_and_return_conditional_losses_8378?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_conv1d_1_layer_call_fn_7503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"?????????????????? 
?2?
B__inference_conv1d_1_layer_call_and_return_conditional_losses_7493?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"?????????????????? 
?2?
(__inference_dropout_1_layer_call_fn_8423
(__inference_dropout_1_layer_call_fn_8418?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_1_layer_call_and_return_conditional_losses_8408
C__inference_dropout_1_layer_call_and_return_conditional_losses_8413?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_conv1d_2_layer_call_fn_7538?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????@
?2?
B__inference_conv1d_2_layer_call_and_return_conditional_losses_7528?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????@
?2?
(__inference_dropout_2_layer_call_fn_8458
(__inference_dropout_2_layer_call_fn_8453?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_2_layer_call_and_return_conditional_losses_8443
C__inference_dropout_2_layer_call_and_return_conditional_losses_8448?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_global_max_pooling1d_layer_call_fn_7551?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_7545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
$__inference_dense_layer_call_fn_8477?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_8468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_activation_layer_call_fn_8487?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_activation_layer_call_and_return_conditional_losses_8482?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_8506?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_8497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_activation_1_layer_call_fn_8516?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_1_layer_call_and_return_conditional_losses_8511?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_8535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_8526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_8548?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_8561?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_8574?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
/B-
"__inference_signature_wrapper_8096input?
__inference__wrapped_model_7433v'(56?@IJ3?0
)?&
$?!
input??????????	
? "1?.
,
dense_2!?
dense_2?????????2?
F__inference_activation_1_layer_call_and_return_conditional_losses_8511Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
+__inference_activation_1_layer_call_fn_8516M0?-
&?#
!?
inputs??????????
? "????????????
D__inference_activation_layer_call_and_return_conditional_losses_8482Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
)__inference_activation_layer_call_fn_8487M0?-
&?#
!?
inputs??????????
? "????????????
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_7790r'(56?@IJ;?8
1?.
$?!
input??????????	
p

 
? "%?"
?
0?????????2
? ?
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_7854r'(56?@IJ;?8
1?.
$?!
input??????????	
p 

 
? "%?"
?
0?????????2
? ?
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_8206s'(56?@IJ<?9
2?/
%?"
inputs??????????	
p

 
? "%?"
?
0?????????2
? ?
K__inference_base_model_simclr_layer_call_and_return_conditional_losses_8295s'(56?@IJ<?9
2?/
%?"
inputs??????????	
p 

 
? "%?"
?
0?????????2
? ?
0__inference_base_model_simclr_layer_call_fn_7948e'(56?@IJ;?8
1?.
$?!
input??????????	
p

 
? "??????????2?
0__inference_base_model_simclr_layer_call_fn_8041e'(56?@IJ;?8
1?.
$?!
input??????????	
p 

 
? "??????????2?
0__inference_base_model_simclr_layer_call_fn_8324f'(56?@IJ<?9
2?/
%?"
inputs??????????	
p

 
? "??????????2?
0__inference_base_model_simclr_layer_call_fn_8353f'(56?@IJ<?9
2?/
%?"
inputs??????????	
p 

 
? "??????????2?
B__inference_conv1d_1_layer_call_and_return_conditional_losses_7493v<?9
2?/
-?*
inputs?????????????????? 
? "2?/
(?%
0??????????????????@
? ?
'__inference_conv1d_1_layer_call_fn_7503i<?9
2?/
-?*
inputs?????????????????? 
? "%?"??????????????????@?
B__inference_conv1d_2_layer_call_and_return_conditional_losses_7528v'(<?9
2?/
-?*
inputs??????????????????@
? "2?/
(?%
0??????????????????`
? ?
'__inference_conv1d_2_layer_call_fn_7538i'(<?9
2?/
-?*
inputs??????????????????@
? "%?"??????????????????`?
@__inference_conv1d_layer_call_and_return_conditional_losses_7458v<?9
2?/
-?*
inputs??????????????????	
? "2?/
(?%
0?????????????????? 
? ?
%__inference_conv1d_layer_call_fn_7468i<?9
2?/
-?*
inputs??????????????????	
? "%?"?????????????????? ?
A__inference_dense_1_layer_call_and_return_conditional_losses_8497^?@0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_1_layer_call_fn_8506Q?@0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_2_layer_call_and_return_conditional_losses_8526]IJ0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????2
? z
&__inference_dense_2_layer_call_fn_8535PIJ0?-
&?#
!?
inputs??????????
? "??????????2?
?__inference_dense_layer_call_and_return_conditional_losses_8468]56/?,
%?"
 ?
inputs?????????`
? "&?#
?
0??????????
? x
$__inference_dense_layer_call_fn_8477P56/?,
%?"
 ?
inputs?????????`
? "????????????
C__inference_dropout_1_layer_call_and_return_conditional_losses_8408d7?4
-?*
$?!
inputs?????????Z@
p
? ")?&
?
0?????????Z@
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_8413d7?4
-?*
$?!
inputs?????????Z@
p 
? ")?&
?
0?????????Z@
? ?
(__inference_dropout_1_layer_call_fn_8418W7?4
-?*
$?!
inputs?????????Z@
p
? "??????????Z@?
(__inference_dropout_1_layer_call_fn_8423W7?4
-?*
$?!
inputs?????????Z@
p 
? "??????????Z@?
C__inference_dropout_2_layer_call_and_return_conditional_losses_8443d7?4
-?*
$?!
inputs?????????S`
p
? ")?&
?
0?????????S`
? ?
C__inference_dropout_2_layer_call_and_return_conditional_losses_8448d7?4
-?*
$?!
inputs?????????S`
p 
? ")?&
?
0?????????S`
? ?
(__inference_dropout_2_layer_call_fn_8453W7?4
-?*
$?!
inputs?????????S`
p
? "??????????S`?
(__inference_dropout_2_layer_call_fn_8458W7?4
-?*
$?!
inputs?????????S`
p 
? "??????????S`?
A__inference_dropout_layer_call_and_return_conditional_losses_8373d7?4
-?*
$?!
inputs?????????i 
p
? ")?&
?
0?????????i 
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_8378d7?4
-?*
$?!
inputs?????????i 
p 
? ")?&
?
0?????????i 
? ?
&__inference_dropout_layer_call_fn_8383W7?4
-?*
$?!
inputs?????????i 
p
? "??????????i ?
&__inference_dropout_layer_call_fn_8388W7?4
-?*
$?!
inputs?????????i 
p 
? "??????????i ?
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_7545wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+
$?!
0??????????????????
? ?
3__inference_global_max_pooling1d_layer_call_fn_7551jE?B
;?8
6?3
inputs'???????????????????????????
? "!???????????????????9
__inference_loss_fn_0_8548?

? 
? "? 9
__inference_loss_fn_1_8561?

? 
? "? 9
__inference_loss_fn_2_8574'?

? 
? "? ?
"__inference_signature_wrapper_8096'(56?@IJ<?9
? 
2?/
-
input$?!
input??????????	"1?.
,
dense_2!?
dense_2?????????2