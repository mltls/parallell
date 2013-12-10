
vc:     file format elf64-x86-64


Disassembly of section .init:

0000000000400858 <_init>:
  400858:	48 83 ec 08          	sub    $0x8,%rsp
  40085c:	e8 cb 05 00 00       	callq  400e2c <call_gmon_start>
  400861:	48 83 c4 08          	add    $0x8,%rsp
  400865:	c3                   	retq   

Disassembly of section .plt:

0000000000400870 <printf@plt-0x10>:
  400870:	ff 35 ca 39 20 00    	pushq  0x2039ca(%rip)        # 604240 <_GLOBAL_OFFSET_TABLE_+0x8>
  400876:	ff 25 cc 39 20 00    	jmpq   *0x2039cc(%rip)        # 604248 <_GLOBAL_OFFSET_TABLE_+0x10>
  40087c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400880 <printf@plt>:
  400880:	ff 25 ca 39 20 00    	jmpq   *0x2039ca(%rip)        # 604250 <_GLOBAL_OFFSET_TABLE_+0x18>
  400886:	68 00 00 00 00       	pushq  $0x0
  40088b:	e9 e0 ff ff ff       	jmpq   400870 <_init+0x18>

0000000000400890 <_ZSt16__throw_bad_castv@plt>:
  400890:	ff 25 c2 39 20 00    	jmpq   *0x2039c2(%rip)        # 604258 <_GLOBAL_OFFSET_TABLE_+0x20>
  400896:	68 01 00 00 00       	pushq  $0x1
  40089b:	e9 d0 ff ff ff       	jmpq   400870 <_init+0x18>

00000000004008a0 <abort@plt>:
  4008a0:	ff 25 ba 39 20 00    	jmpq   *0x2039ba(%rip)        # 604260 <_GLOBAL_OFFSET_TABLE_+0x28>
  4008a6:	68 02 00 00 00       	pushq  $0x2
  4008ab:	e9 c0 ff ff ff       	jmpq   400870 <_init+0x18>

00000000004008b0 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>:
  4008b0:	ff 25 b2 39 20 00    	jmpq   *0x2039b2(%rip)        # 604268 <_GLOBAL_OFFSET_TABLE_+0x30>
  4008b6:	68 03 00 00 00       	pushq  $0x3
  4008bb:	e9 b0 ff ff ff       	jmpq   400870 <_init+0x18>

00000000004008c0 <_ZNSt8ios_base4InitC1Ev@plt>:
  4008c0:	ff 25 aa 39 20 00    	jmpq   *0x2039aa(%rip)        # 604270 <_GLOBAL_OFFSET_TABLE_+0x38>
  4008c6:	68 04 00 00 00       	pushq  $0x4
  4008cb:	e9 a0 ff ff ff       	jmpq   400870 <_init+0x18>

00000000004008d0 <__libc_start_main@plt>:
  4008d0:	ff 25 a2 39 20 00    	jmpq   *0x2039a2(%rip)        # 604278 <_GLOBAL_OFFSET_TABLE_+0x40>
  4008d6:	68 05 00 00 00       	pushq  $0x5
  4008db:	e9 90 ff ff ff       	jmpq   400870 <_init+0x18>

00000000004008e0 <__cxa_atexit@plt>:
  4008e0:	ff 25 9a 39 20 00    	jmpq   *0x20399a(%rip)        # 604280 <_GLOBAL_OFFSET_TABLE_+0x48>
  4008e6:	68 06 00 00 00       	pushq  $0x6
  4008eb:	e9 80 ff ff ff       	jmpq   400870 <_init+0x18>

00000000004008f0 <_ZNSt8ios_base4InitD1Ev@plt>:
  4008f0:	ff 25 92 39 20 00    	jmpq   *0x203992(%rip)        # 604288 <_GLOBAL_OFFSET_TABLE_+0x50>
  4008f6:	68 07 00 00 00       	pushq  $0x7
  4008fb:	e9 70 ff ff ff       	jmpq   400870 <_init+0x18>

0000000000400900 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>:
  400900:	ff 25 8a 39 20 00    	jmpq   *0x20398a(%rip)        # 604290 <_GLOBAL_OFFSET_TABLE_+0x58>
  400906:	68 08 00 00 00       	pushq  $0x8
  40090b:	e9 60 ff ff ff       	jmpq   400870 <_init+0x18>

0000000000400910 <_ZNSo5flushEv@plt>:
  400910:	ff 25 82 39 20 00    	jmpq   *0x203982(%rip)        # 604298 <_GLOBAL_OFFSET_TABLE_+0x60>
  400916:	68 09 00 00 00       	pushq  $0x9
  40091b:	e9 50 ff ff ff       	jmpq   400870 <_init+0x18>

0000000000400920 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>:
  400920:	ff 25 7a 39 20 00    	jmpq   *0x20397a(%rip)        # 6042a0 <_GLOBAL_OFFSET_TABLE_+0x68>
  400926:	68 0a 00 00 00       	pushq  $0xa
  40092b:	e9 40 ff ff ff       	jmpq   400870 <_init+0x18>

0000000000400930 <_ZNSo9_M_insertImEERSoT_@plt>:
  400930:	ff 25 72 39 20 00    	jmpq   *0x203972(%rip)        # 6042a8 <_GLOBAL_OFFSET_TABLE_+0x70>
  400936:	68 0b 00 00 00       	pushq  $0xb
  40093b:	e9 30 ff ff ff       	jmpq   400870 <_init+0x18>

0000000000400940 <_ZNSo9_M_insertIdEERSoT_@plt>:
  400940:	ff 25 6a 39 20 00    	jmpq   *0x20396a(%rip)        # 6042b0 <_GLOBAL_OFFSET_TABLE_+0x78>
  400946:	68 0c 00 00 00       	pushq  $0xc
  40094b:	e9 20 ff ff ff       	jmpq   400870 <_init+0x18>

0000000000400950 <_ZNSo3putEc@plt>:
  400950:	ff 25 62 39 20 00    	jmpq   *0x203962(%rip)        # 6042b8 <_GLOBAL_OFFSET_TABLE_+0x80>
  400956:	68 0d 00 00 00       	pushq  $0xd
  40095b:	e9 10 ff ff ff       	jmpq   400870 <_init+0x18>

Disassembly of section .text:

0000000000400960 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.part.15>:
  400960:	50                   	push   %rax
  400961:	48 8b 07             	mov    (%rdi),%rax
  400964:	48 03 78 e8          	add    -0x18(%rax),%rdi
  400968:	8b 77 20             	mov    0x20(%rdi),%esi
  40096b:	83 ce 01             	or     $0x1,%esi
  40096e:	e8 ad ff ff ff       	callq  400920 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
  400973:	5a                   	pop    %rdx
  400974:	c3                   	retq   
  400975:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40097c:	00 00 00 
  40097f:	90                   	nop

0000000000400980 <main>:
  400980:	55                   	push   %rbp
  400981:	48 89 e5             	mov    %rsp,%rbp
  400984:	41 55                	push   %r13
  400986:	41 54                	push   %r12
  400988:	53                   	push   %rbx
  400989:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40098d:	48 81 ec d0 3e 00 00 	sub    $0x3ed0,%rsp
  400994:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
  400999:	e8 72 05 00 00       	callq  400f10 <_ZN2Vc2v06Common6MemoryINS0_3AVX6VectorIfEELm1000ELm0ELb1EEC1Ev>
  40099e:	48 8d bc 24 f0 0f 00 	lea    0xff0(%rsp),%rdi
  4009a5:	00 
  4009a6:	e8 65 05 00 00       	callq  400f10 <_ZN2Vc2v06Common6MemoryINS0_3AVX6VectorIfEELm1000ELm0ELb1EEC1Ev>
  4009ab:	48 8d bc 24 90 1f 00 	lea    0x1f90(%rsp),%rdi
  4009b2:	00 
  4009b3:	e8 58 05 00 00       	callq  400f10 <_ZN2Vc2v06Common6MemoryINS0_3AVX6VectorIfEELm1000ELm0ELb1EEC1Ev>
  4009b8:	48 8d bc 24 30 2f 00 	lea    0x2f30(%rsp),%rdi
  4009bf:	00 
  4009c0:	e8 4b 05 00 00       	callq  400f10 <_ZN2Vc2v06Common6MemoryINS0_3AVX6VectorIfEELm1000ELm0ELb1EEC1Ev>
  4009c5:	c5 fd 6f 0d b3 1e 00 	vmovdqa 0x1eb3(%rip),%ymm1        # 402880 <__dso_handle+0x38>
  4009cc:	00 
  4009cd:	31 c0                	xor    %eax,%eax
  4009cf:	c5 fd 6f 05 c9 1e 00 	vmovdqa 0x1ec9(%rip),%ymm0        # 4028a0 <__dso_handle+0x58>
  4009d6:	00 
  4009d7:	c4 e3 7d 19 c9 01    	vextractf128 $0x1,%ymm1,%xmm1
  4009dd:	c4 e2 7d 18 15 26 2d 	vbroadcastss 0x2d26(%rip),%ymm2        # 40370c <_ZN2Vc2v03AVX9c_general8oneFloatE>
  4009e4:	00 00 
  4009e6:	c4 e3 7d 19 c0 01    	vextractf128 $0x1,%ymm0,%xmm0
  4009ec:	c5 fc 28 3d cc 1e 00 	vmovaps 0x1ecc(%rip),%ymm7        # 4028c0 <__dso_handle+0x78>
  4009f3:	00 
  4009f4:	0f 1f 40 00          	nopl   0x0(%rax)
  4009f8:	c5 fd 6f 25 20 39 20 	vmovdqa 0x203920(%rip),%ymm4        # 604320 <_ZN2Vc2v06Common11RandomStateE+0x20>
  4009ff:	00 
  400a00:	c5 fd 6f 35 f8 38 20 	vmovdqa 0x2038f8(%rip),%ymm6        # 604300 <_ZN2Vc2v06Common11RandomStateE>
  400a07:	00 
  400a08:	c4 e3 7d 19 e3 01    	vextractf128 $0x1,%ymm4,%xmm3
  400a0e:	c4 62 59 40 c1       	vpmulld %xmm1,%xmm4,%xmm8
  400a13:	c5 a9 72 d4 10       	vpsrld $0x10,%xmm4,%xmm10
  400a18:	c4 e2 61 40 e9       	vpmulld %xmm1,%xmm3,%xmm5
  400a1d:	c5 e1 72 d3 10       	vpsrld $0x10,%xmm3,%xmm3
  400a22:	c4 63 3d 18 c5 01    	vinsertf128 $0x1,%xmm5,%ymm8,%ymm8
  400a28:	c4 63 2d 18 d3 01    	vinsertf128 $0x1,%xmm3,%ymm10,%ymm10
  400a2e:	c4 e3 7d 19 f3 01    	vextractf128 $0x1,%ymm6,%xmm3
  400a34:	c4 43 7d 19 c1 01    	vextractf128 $0x1,%ymm8,%xmm9
  400a3a:	c5 79 7f c5          	vmovdqa %xmm8,%xmm5
  400a3e:	c4 62 49 40 c1       	vpmulld %xmm1,%xmm6,%xmm8
  400a43:	c5 c9 72 d6 02       	vpsrld $0x2,%xmm6,%xmm6
  400a48:	c4 e2 61 40 e1       	vpmulld %xmm1,%xmm3,%xmm4
  400a4d:	c5 e1 72 d3 02       	vpsrld $0x2,%xmm3,%xmm3
  400a52:	c4 e3 4d 18 db 01    	vinsertf128 $0x1,%xmm3,%ymm6,%ymm3
  400a58:	c5 d1 fe e8          	vpaddd %xmm0,%xmm5,%xmm5
  400a5c:	c5 31 fe c8          	vpaddd %xmm0,%xmm9,%xmm9
  400a60:	c4 c3 55 18 e9 01    	vinsertf128 $0x1,%xmm9,%ymm5,%ymm5
  400a66:	c4 63 3d 18 c4 01    	vinsertf128 $0x1,%xmm4,%ymm8,%ymm8
  400a6c:	c5 ec 56 db          	vorps  %ymm3,%ymm2,%ymm3
  400a70:	c4 e2 51 40 f1       	vpmulld %xmm1,%xmm5,%xmm6
  400a75:	c4 43 7d 19 c3 01    	vextractf128 $0x1,%ymm8,%xmm11
  400a7b:	c5 e4 5c da          	vsubps %ymm2,%ymm3,%ymm3
  400a7f:	c5 79 7f c4          	vmovdqa %xmm8,%xmm4
  400a83:	c5 21 fe d8          	vpaddd %xmm0,%xmm11,%xmm11
  400a87:	c5 d9 fe e0          	vpaddd %xmm0,%xmm4,%xmm4
  400a8b:	c4 c3 5d 18 e3 01    	vinsertf128 $0x1,%xmm11,%ymm4,%ymm4
  400a91:	c5 e4 58 db          	vaddps %ymm3,%ymm3,%ymm3
  400a95:	c4 c1 5c 57 e2       	vxorps %ymm10,%ymm4,%ymm4
  400a9a:	c5 e4 5c df          	vsubps %ymm7,%ymm3,%ymm3
  400a9e:	c5 fc 29 5c 04 50    	vmovaps %ymm3,0x50(%rsp,%rax,1)
  400aa4:	c4 e3 7d 19 eb 01    	vextractf128 $0x1,%ymm5,%xmm3
  400aaa:	c4 62 61 40 c1       	vpmulld %xmm1,%xmm3,%xmm8
  400aaf:	c4 c3 4d 18 f0 01    	vinsertf128 $0x1,%xmm8,%ymm6,%ymm6
  400ab5:	c5 e1 72 d3 10       	vpsrld $0x10,%xmm3,%xmm3
  400aba:	c4 c3 7d 19 f0 01    	vextractf128 $0x1,%ymm6,%xmm8
  400ac0:	c5 c9 fe f0          	vpaddd %xmm0,%xmm6,%xmm6
  400ac4:	c5 39 fe c0          	vpaddd %xmm0,%xmm8,%xmm8
  400ac8:	c4 c3 4d 18 f0 01    	vinsertf128 $0x1,%xmm8,%ymm6,%ymm6
  400ace:	c5 fd 7f 35 4a 38 20 	vmovdqa %ymm6,0x20384a(%rip)        # 604320 <_ZN2Vc2v06Common11RandomStateE+0x20>
  400ad5:	00 
  400ad6:	c5 c9 72 d5 10       	vpsrld $0x10,%xmm5,%xmm6
  400adb:	c4 e3 4d 18 f3 01    	vinsertf128 $0x1,%xmm3,%ymm6,%ymm6
  400ae1:	c4 e3 7d 19 e3 01    	vextractf128 $0x1,%ymm4,%xmm3
  400ae7:	c4 e2 59 40 e9       	vpmulld %xmm1,%xmm4,%xmm5
  400aec:	c5 d9 72 d4 02       	vpsrld $0x2,%xmm4,%xmm4
  400af1:	c4 62 61 40 c1       	vpmulld %xmm1,%xmm3,%xmm8
  400af6:	c5 e1 72 d3 02       	vpsrld $0x2,%xmm3,%xmm3
  400afb:	c4 e3 5d 18 db 01    	vinsertf128 $0x1,%xmm3,%ymm4,%ymm3
  400b01:	c4 c3 55 18 e8 01    	vinsertf128 $0x1,%xmm8,%ymm5,%ymm5
  400b07:	c5 ec 56 db          	vorps  %ymm3,%ymm2,%ymm3
  400b0b:	c4 c3 7d 19 e8 01    	vextractf128 $0x1,%ymm5,%xmm8
  400b11:	c5 d1 fe e8          	vpaddd %xmm0,%xmm5,%xmm5
  400b15:	c5 e4 5c da          	vsubps %ymm2,%ymm3,%ymm3
  400b19:	c5 39 fe c0          	vpaddd %xmm0,%xmm8,%xmm8
  400b1d:	c4 c3 55 18 e8 01    	vinsertf128 $0x1,%xmm8,%ymm5,%ymm5
  400b23:	c5 d4 57 ee          	vxorps %ymm6,%ymm5,%ymm5
  400b27:	c5 e4 58 db          	vaddps %ymm3,%ymm3,%ymm3
  400b2b:	c5 fd 7f 2d cd 37 20 	vmovdqa %ymm5,0x2037cd(%rip)        # 604300 <_ZN2Vc2v06Common11RandomStateE>
  400b32:	00 
  400b33:	c5 e4 5c df          	vsubps %ymm7,%ymm3,%ymm3
  400b37:	c5 fc 29 9c 04 f0 0f 	vmovaps %ymm3,0xff0(%rsp,%rax,1)
  400b3e:	00 00 
  400b40:	48 83 c0 20          	add    $0x20,%rax
  400b44:	48 3d a0 0f 00 00    	cmp    $0xfa0,%rax
  400b4a:	0f 85 a8 fe ff ff    	jne    4009f8 <main+0x78>
  400b50:	31 db                	xor    %ebx,%ebx
  400b52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400b58:	c5 fc 28 44 1c 50    	vmovaps 0x50(%rsp,%rbx,1),%ymm0
  400b5e:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
  400b63:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  400b68:	c5 fc 28 8c 1c f0 0f 	vmovaps 0xff0(%rsp,%rbx,1),%ymm1
  400b6f:	00 00 
  400b71:	c5 fc 29 44 24 10    	vmovaps %ymm0,0x10(%rsp)
  400b77:	c5 fc 59 c0          	vmulps %ymm0,%ymm0,%ymm0
  400b7b:	c5 fc 29 4c 24 30    	vmovaps %ymm1,0x30(%rsp)
  400b81:	c5 f4 59 c9          	vmulps %ymm1,%ymm1,%ymm1
  400b85:	c5 f4 58 c0          	vaddps %ymm0,%ymm1,%ymm0
  400b89:	c5 fc 51 c0          	vsqrtps %ymm0,%ymm0
  400b8d:	c5 fc 29 84 1c 90 1f 	vmovaps %ymm0,0x1f90(%rsp,%rbx,1)
  400b94:	00 00 
  400b96:	c5 f8 77             	vzeroupper 
  400b99:	e8 d2 19 00 00       	callq  402570 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE5atan2IfEENS0_3AVX6VectorIT_EERKSB_SD_>
  400b9e:	c5 fc 59 05 3a 1d 00 	vmulps 0x1d3a(%rip),%ymm0,%ymm0        # 4028e0 <__dso_handle+0x98>
  400ba5:	00 
  400ba6:	c5 fc c2 0d 51 1d 00 	vcmpltps 0x1d51(%rip),%ymm0,%ymm1        # 402900 <__dso_handle+0xb8>
  400bad:	00 01 
  400baf:	c5 f4 54 0d 69 1d 00 	vandps 0x1d69(%rip),%ymm1,%ymm1        # 402920 <__dso_handle+0xd8>
  400bb6:	00 
  400bb7:	c5 fc 58 c1          	vaddps %ymm1,%ymm0,%ymm0
  400bbb:	c5 fc 29 84 1c 30 2f 	vmovaps %ymm0,0x2f30(%rsp,%rbx,1)
  400bc2:	00 00 
  400bc4:	48 83 c3 20          	add    $0x20,%rbx
  400bc8:	48 81 fb a0 0f 00 00 	cmp    $0xfa0,%rbx
  400bcf:	75 87                	jne    400b58 <main+0x1d8>
  400bd1:	66 31 db             	xor    %bx,%bx
  400bd4:	c5 f8 77             	vzeroupper 
  400bd7:	eb 31                	jmp    400c0a <main+0x28a>
  400bd9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400be0:	41 0f b6 44 24 43    	movzbl 0x43(%r12),%eax
  400be6:	0f be f0             	movsbl %al,%esi
  400be9:	4c 89 ef             	mov    %r13,%rdi
  400bec:	48 83 c3 01          	add    $0x1,%rbx
  400bf0:	e8 5b fd ff ff       	callq  400950 <_ZNSo3putEc@plt>
  400bf5:	48 89 c7             	mov    %rax,%rdi
  400bf8:	e8 13 fd ff ff       	callq  400910 <_ZNSo5flushEv@plt>
  400bfd:	48 81 fb e8 03 00 00 	cmp    $0x3e8,%rbx
  400c04:	0f 84 73 01 00 00    	je     400d7d <main+0x3fd>
  400c0a:	48 8b 05 2f 37 20 00 	mov    0x20372f(%rip),%rax        # 604340 <__TMC_END__>
  400c11:	48 89 de             	mov    %rbx,%rsi
  400c14:	bf 40 43 60 00       	mov    $0x604340,%edi
  400c19:	48 8b 40 e8          	mov    -0x18(%rax),%rax
  400c1d:	48 c7 80 50 43 60 00 	movq   $0x3,0x604350(%rax)
  400c24:	03 00 00 00 
  400c28:	e8 03 fd ff ff       	callq  400930 <_ZNSo9_M_insertImEERSoT_@plt>
  400c2d:	ba 02 00 00 00       	mov    $0x2,%edx
  400c32:	be 50 28 40 00       	mov    $0x402850,%esi
  400c37:	48 89 c7             	mov    %rax,%rdi
  400c3a:	e8 c1 fc ff ff       	callq  400900 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  400c3f:	c5 fa 10 44 9c 50    	vmovss 0x50(%rsp,%rbx,4),%xmm0
  400c45:	48 8b 05 f4 36 20 00 	mov    0x2036f4(%rip),%rax        # 604340 <__TMC_END__>
  400c4c:	c5 f8 14 c0          	vunpcklps %xmm0,%xmm0,%xmm0
  400c50:	bf 40 43 60 00       	mov    $0x604340,%edi
  400c55:	c5 fa 10 bc 9c f0 0f 	vmovss 0xff0(%rsp,%rbx,4),%xmm7
  400c5c:	00 00 
  400c5e:	48 8b 40 e8          	mov    -0x18(%rax),%rax
  400c62:	c5 fa 11 7c 24 0c    	vmovss %xmm7,0xc(%rsp)
  400c68:	c5 f8 5a c0          	vcvtps2pd %xmm0,%xmm0
  400c6c:	48 c7 80 50 43 60 00 	movq   $0xa,0x604350(%rax)
  400c73:	0a 00 00 00 
  400c77:	e8 c4 fc ff ff       	callq  400940 <_ZNSo9_M_insertIdEERSoT_@plt>
  400c7c:	ba 02 00 00 00       	mov    $0x2,%edx
  400c81:	49 89 c4             	mov    %rax,%r12
  400c84:	be 53 28 40 00       	mov    $0x402853,%esi
  400c89:	48 89 c7             	mov    %rax,%rdi
  400c8c:	e8 6f fc ff ff       	callq  400900 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  400c91:	49 8b 04 24          	mov    (%r12),%rax
  400c95:	c5 fa 10 44 24 0c    	vmovss 0xc(%rsp),%xmm0
  400c9b:	4c 89 e7             	mov    %r12,%rdi
  400c9e:	c5 f8 5a c0          	vcvtps2pd %xmm0,%xmm0
  400ca2:	48 8b 40 e8          	mov    -0x18(%rax),%rax
  400ca6:	49 c7 44 04 10 0a 00 	movq   $0xa,0x10(%r12,%rax,1)
  400cad:	00 00 
  400caf:	e8 8c fc ff ff       	callq  400940 <_ZNSo9_M_insertIdEERSoT_@plt>
  400cb4:	ba 04 00 00 00       	mov    $0x4,%edx
  400cb9:	be 56 28 40 00       	mov    $0x402856,%esi
  400cbe:	48 89 c7             	mov    %rax,%rdi
  400cc1:	e8 3a fc ff ff       	callq  400900 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  400cc6:	c5 fa 10 84 9c 90 1f 	vmovss 0x1f90(%rsp,%rbx,4),%xmm0
  400ccd:	00 00 
  400ccf:	48 8b 05 6a 36 20 00 	mov    0x20366a(%rip),%rax        # 604340 <__TMC_END__>
  400cd6:	c5 f8 14 c0          	vunpcklps %xmm0,%xmm0,%xmm0
  400cda:	bf 40 43 60 00       	mov    $0x604340,%edi
  400cdf:	c5 fa 10 bc 9c 30 2f 	vmovss 0x2f30(%rsp,%rbx,4),%xmm7
  400ce6:	00 00 
  400ce8:	48 8b 40 e8          	mov    -0x18(%rax),%rax
  400cec:	c5 fa 11 7c 24 0c    	vmovss %xmm7,0xc(%rsp)
  400cf2:	c5 f8 5a c0          	vcvtps2pd %xmm0,%xmm0
  400cf6:	48 c7 80 50 43 60 00 	movq   $0xa,0x604350(%rax)
  400cfd:	0a 00 00 00 
  400d01:	e8 3a fc ff ff       	callq  400940 <_ZNSo9_M_insertIdEERSoT_@plt>
  400d06:	ba 02 00 00 00       	mov    $0x2,%edx
  400d0b:	49 89 c4             	mov    %rax,%r12
  400d0e:	be 53 28 40 00       	mov    $0x402853,%esi
  400d13:	48 89 c7             	mov    %rax,%rdi
  400d16:	e8 e5 fb ff ff       	callq  400900 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  400d1b:	49 8b 04 24          	mov    (%r12),%rax
  400d1f:	c5 fa 10 44 24 0c    	vmovss 0xc(%rsp),%xmm0
  400d25:	4c 89 e7             	mov    %r12,%rdi
  400d28:	c5 f8 5a c0          	vcvtps2pd %xmm0,%xmm0
  400d2c:	48 8b 40 e8          	mov    -0x18(%rax),%rax
  400d30:	49 c7 44 04 10 0a 00 	movq   $0xa,0x10(%r12,%rax,1)
  400d37:	00 00 
  400d39:	e8 02 fc ff ff       	callq  400940 <_ZNSo9_M_insertIdEERSoT_@plt>
  400d3e:	49 89 c5             	mov    %rax,%r13
  400d41:	48 8b 00             	mov    (%rax),%rax
  400d44:	48 8b 40 e8          	mov    -0x18(%rax),%rax
  400d48:	4d 8b a4 05 f0 00 00 	mov    0xf0(%r13,%rax,1),%r12
  400d4f:	00 
  400d50:	4d 85 e4             	test   %r12,%r12
  400d53:	74 35                	je     400d8a <main+0x40a>
  400d55:	41 80 7c 24 38 00    	cmpb   $0x0,0x38(%r12)
  400d5b:	0f 85 7f fe ff ff    	jne    400be0 <main+0x260>
  400d61:	4c 89 e7             	mov    %r12,%rdi
  400d64:	e8 47 fb ff ff       	callq  4008b0 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
  400d69:	49 8b 04 24          	mov    (%r12),%rax
  400d6d:	be 0a 00 00 00       	mov    $0xa,%esi
  400d72:	4c 89 e7             	mov    %r12,%rdi
  400d75:	ff 50 30             	callq  *0x30(%rax)
  400d78:	e9 69 fe ff ff       	jmpq   400be6 <main+0x266>
  400d7d:	48 8d 65 e8          	lea    -0x18(%rbp),%rsp
  400d81:	31 c0                	xor    %eax,%eax
  400d83:	5b                   	pop    %rbx
  400d84:	41 5c                	pop    %r12
  400d86:	41 5d                	pop    %r13
  400d88:	5d                   	pop    %rbp
  400d89:	c3                   	retq   
  400d8a:	e8 01 fb ff ff       	callq  400890 <_ZSt16__throw_bad_castv@plt>
  400d8f:	90                   	nop

0000000000400d90 <_GLOBAL__sub_I_main>:
  400d90:	48 83 ec 08          	sub    $0x8,%rsp
  400d94:	bf 54 44 60 00       	mov    $0x604454,%edi
  400d99:	e8 22 fb ff ff       	callq  4008c0 <_ZNSt8ios_base4InitC1Ev@plt>
  400d9e:	ba 48 28 40 00       	mov    $0x402848,%edx
  400da3:	be 54 44 60 00       	mov    $0x604454,%esi
  400da8:	bf f0 08 40 00       	mov    $0x4008f0,%edi
  400dad:	e8 2e fb ff ff       	callq  4008e0 <__cxa_atexit@plt>
  400db2:	ba 5b 28 40 00       	mov    $0x40285b,%edx
  400db7:	be 8f 63 00 00       	mov    $0x638f,%esi
  400dbc:	bf 04 00 00 00       	mov    $0x4,%edi
  400dc1:	48 83 c4 08          	add    $0x8,%rsp
  400dc5:	e9 76 01 00 00       	jmpq   400f40 <_ZN2Vc2v06Common15checkLibraryAbiEjjPKc>
  400dca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400dd0 <_GLOBAL__sub_I_trigonometric_AVX.cpp>:
  400dd0:	48 8d 3d 81 36 20 00 	lea    0x203681(%rip),%rdi        # 604458 <_ZStL8__ioinit>
  400dd7:	48 83 ec 08          	sub    $0x8,%rsp
  400ddb:	e8 e0 fa ff ff       	callq  4008c0 <_ZNSt8ios_base4InitC1Ev@plt>
  400de0:	48 8b 3d 49 34 20 00 	mov    0x203449(%rip),%rdi        # 604230 <_DYNAMIC+0x208>
  400de7:	48 8d 15 5a 1a 00 00 	lea    0x1a5a(%rip),%rdx        # 402848 <__dso_handle>
  400dee:	48 8d 35 63 36 20 00 	lea    0x203663(%rip),%rsi        # 604458 <_ZStL8__ioinit>
  400df5:	48 83 c4 08          	add    $0x8,%rsp
  400df9:	e9 e2 fa ff ff       	jmpq   4008e0 <__cxa_atexit@plt>
  400dfe:	66 90                	xchg   %ax,%ax

0000000000400e00 <_start>:
  400e00:	31 ed                	xor    %ebp,%ebp
  400e02:	49 89 d1             	mov    %rdx,%r9
  400e05:	5e                   	pop    %rsi
  400e06:	48 89 e2             	mov    %rsp,%rdx
  400e09:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  400e0d:	50                   	push   %rax
  400e0e:	54                   	push   %rsp
  400e0f:	49 c7 c0 90 27 40 00 	mov    $0x402790,%r8
  400e16:	48 c7 c1 a0 27 40 00 	mov    $0x4027a0,%rcx
  400e1d:	48 c7 c7 80 09 40 00 	mov    $0x400980,%rdi
  400e24:	e8 a7 fa ff ff       	callq  4008d0 <__libc_start_main@plt>
  400e29:	f4                   	hlt    
  400e2a:	90                   	nop
  400e2b:	90                   	nop

0000000000400e2c <call_gmon_start>:
  400e2c:	48 83 ec 08          	sub    $0x8,%rsp
  400e30:	48 8b 05 f1 33 20 00 	mov    0x2033f1(%rip),%rax        # 604228 <_DYNAMIC+0x200>
  400e37:	48 85 c0             	test   %rax,%rax
  400e3a:	74 02                	je     400e3e <call_gmon_start+0x12>
  400e3c:	ff d0                	callq  *%rax
  400e3e:	48 83 c4 08          	add    $0x8,%rsp
  400e42:	c3                   	retq   
  400e43:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e4a:	00 00 00 
  400e4d:	0f 1f 00             	nopl   (%rax)

0000000000400e50 <deregister_tm_clones>:
  400e50:	b8 47 43 60 00       	mov    $0x604347,%eax
  400e55:	55                   	push   %rbp
  400e56:	48 2d 40 43 60 00    	sub    $0x604340,%rax
  400e5c:	48 83 f8 0e          	cmp    $0xe,%rax
  400e60:	48 89 e5             	mov    %rsp,%rbp
  400e63:	77 02                	ja     400e67 <deregister_tm_clones+0x17>
  400e65:	5d                   	pop    %rbp
  400e66:	c3                   	retq   
  400e67:	b8 00 00 00 00       	mov    $0x0,%eax
  400e6c:	48 85 c0             	test   %rax,%rax
  400e6f:	74 f4                	je     400e65 <deregister_tm_clones+0x15>
  400e71:	5d                   	pop    %rbp
  400e72:	bf 40 43 60 00       	mov    $0x604340,%edi
  400e77:	ff e0                	jmpq   *%rax
  400e79:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400e80 <register_tm_clones>:
  400e80:	b8 40 43 60 00       	mov    $0x604340,%eax
  400e85:	55                   	push   %rbp
  400e86:	48 2d 40 43 60 00    	sub    $0x604340,%rax
  400e8c:	48 c1 f8 03          	sar    $0x3,%rax
  400e90:	48 89 e5             	mov    %rsp,%rbp
  400e93:	48 89 c2             	mov    %rax,%rdx
  400e96:	48 c1 ea 3f          	shr    $0x3f,%rdx
  400e9a:	48 01 d0             	add    %rdx,%rax
  400e9d:	48 d1 f8             	sar    %rax
  400ea0:	75 02                	jne    400ea4 <register_tm_clones+0x24>
  400ea2:	5d                   	pop    %rbp
  400ea3:	c3                   	retq   
  400ea4:	ba 00 00 00 00       	mov    $0x0,%edx
  400ea9:	48 85 d2             	test   %rdx,%rdx
  400eac:	74 f4                	je     400ea2 <register_tm_clones+0x22>
  400eae:	5d                   	pop    %rbp
  400eaf:	48 89 c6             	mov    %rax,%rsi
  400eb2:	bf 40 43 60 00       	mov    $0x604340,%edi
  400eb7:	ff e2                	jmpq   *%rdx
  400eb9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400ec0 <__do_global_dtors_aux>:
  400ec0:	80 3d 89 35 20 00 00 	cmpb   $0x0,0x203589(%rip)        # 604450 <completed.6303>
  400ec7:	75 11                	jne    400eda <__do_global_dtors_aux+0x1a>
  400ec9:	55                   	push   %rbp
  400eca:	48 89 e5             	mov    %rsp,%rbp
  400ecd:	e8 7e ff ff ff       	callq  400e50 <deregister_tm_clones>
  400ed2:	5d                   	pop    %rbp
  400ed3:	c6 05 76 35 20 00 01 	movb   $0x1,0x203576(%rip)        # 604450 <completed.6303>
  400eda:	f3 c3                	repz retq 
  400edc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ee0 <frame_dummy>:
  400ee0:	48 83 3d 38 31 20 00 	cmpq   $0x0,0x203138(%rip)        # 604020 <__JCR_END__>
  400ee7:	00 
  400ee8:	74 1e                	je     400f08 <frame_dummy+0x28>
  400eea:	b8 00 00 00 00       	mov    $0x0,%eax
  400eef:	48 85 c0             	test   %rax,%rax
  400ef2:	74 14                	je     400f08 <frame_dummy+0x28>
  400ef4:	55                   	push   %rbp
  400ef5:	bf 20 40 60 00       	mov    $0x604020,%edi
  400efa:	48 89 e5             	mov    %rsp,%rbp
  400efd:	ff d0                	callq  *%rax
  400eff:	5d                   	pop    %rbp
  400f00:	e9 7b ff ff ff       	jmpq   400e80 <register_tm_clones>
  400f05:	0f 1f 00             	nopl   (%rax)
  400f08:	e9 73 ff ff ff       	jmpq   400e80 <register_tm_clones>
  400f0d:	0f 1f 00             	nopl   (%rax)

0000000000400f10 <_ZN2Vc2v06Common6MemoryINS0_3AVX6VectorIfEELm1000ELm0ELb1EEC1Ev>:
  400f10:	55                   	push   %rbp
  400f11:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400f15:	48 89 e5             	mov    %rsp,%rbp
  400f18:	c5 fc 29 87 80 0f 00 	vmovaps %ymm0,0xf80(%rdi)
  400f1f:	00 
  400f20:	c5 f8 77             	vzeroupper 
  400f23:	5d                   	pop    %rbp
  400f24:	c3                   	retq   
  400f25:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f2c:	00 00 00 
  400f2f:	90                   	nop

0000000000400f30 <_ZN2Vc2v06Common8Warnings25_operator_bracket_warningEv>:
  400f30:	f3 c3                	repz retq 
  400f32:	66 66 66 66 66 2e 0f 	data32 data32 data32 data32 nopw %cs:0x0(%rax,%rax,1)
  400f39:	1f 84 00 00 00 00 00 

0000000000400f40 <_ZN2Vc2v06Common15checkLibraryAbiEjjPKc>:
  400f40:	48 83 ec 08          	sub    $0x8,%rsp
  400f44:	83 ff 04             	cmp    $0x4,%edi
  400f47:	75 0d                	jne    400f56 <_ZN2Vc2v06Common15checkLibraryAbiEjjPKc+0x16>
  400f49:	81 fe 8f 63 00 00    	cmp    $0x638f,%esi
  400f4f:	77 05                	ja     400f56 <_ZN2Vc2v06Common15checkLibraryAbiEjjPKc+0x16>
  400f51:	48 83 c4 08          	add    $0x8,%rsp
  400f55:	c3                   	retq   
  400f56:	48 8d 35 63 1a 00 00 	lea    0x1a63(%rip),%rsi        # 4029c0 <_ZN2Vc2v06CommonL15LIBRARY_VERSIONE>
  400f5d:	48 8d 3d dc 19 00 00 	lea    0x19dc(%rip),%rdi        # 402940 <__dso_handle+0xf8>
  400f64:	31 c0                	xor    %eax,%eax
  400f66:	e8 15 f9 ff ff       	callq  400880 <printf@plt>
  400f6b:	e8 30 f9 ff ff       	callq  4008a0 <abort@plt>

0000000000400f70 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE3sinIdEENS0_3AVX6VectorIT_EERKSB_>:
  400f70:	48 8d 05 a5 27 00 00 	lea    0x27a5(%rip),%rax        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  400f77:	c5 fd 28 07          	vmovapd (%rdi),%ymm0
  400f7b:	c5 c1 57 ff          	vxorpd %xmm7,%xmm7,%xmm7
  400f7f:	48 8d 15 ba 2a 00 00 	lea    0x2aba(%rip),%rdx        # 403a40 <_ZN2Vc2v03AVX18_IndexesFromZero32E>
  400f86:	c5 e9 ef d2          	vpxor  %xmm2,%xmm2,%xmm2
  400f8a:	c4 e2 7d 19 30       	vbroadcastsd (%rax),%ymm6
  400f8f:	48 8d 05 aa 28 00 00 	lea    0x28aa(%rip),%rax        # 403840 <_ZN2Vc2v03AVX6c_trigIdE4dataE>
  400f96:	c5 fd c2 ff 01       	vcmpltpd %ymm7,%ymm0,%ymm7
  400f9b:	c4 e2 7d 18 62 04    	vbroadcastss 0x4(%rdx),%ymm4
  400fa1:	48 8d 15 f8 25 00 00 	lea    0x25f8(%rip),%rdx        # 4035a0 <_ZN2Vc2v06Common10AllBitsSetE>
  400fa8:	c5 fd 54 f6          	vandpd %ymm6,%ymm0,%ymm6
  400fac:	c4 e2 7d 19 18       	vbroadcastsd (%rax),%ymm3
  400fb1:	c4 e2 7d 19 48 20    	vbroadcastsd 0x20(%rax),%ymm1
  400fb7:	c4 e2 7d 19 40 28    	vbroadcastsd 0x28(%rax),%ymm0
  400fbd:	c4 62 7d 19 98 90 00 	vbroadcastsd 0x90(%rax),%ymm11
  400fc4:	00 00 
  400fc6:	c4 62 7d 19 90 88 00 	vbroadcastsd 0x88(%rax),%ymm10
  400fcd:	00 00 
  400fcf:	c4 62 7d 19 88 80 00 	vbroadcastsd 0x80(%rax),%ymm9
  400fd6:	00 00 
  400fd8:	c5 cd 5e db          	vdivpd %ymm3,%ymm6,%ymm3
  400fdc:	c4 e3 7d 09 db 03    	vroundpd $0x3,%ymm3,%ymm3
  400fe2:	c5 e5 59 c9          	vmulpd %ymm1,%ymm3,%ymm1
  400fe6:	c4 e3 7d 09 c9 03    	vroundpd $0x3,%ymm1,%ymm1
  400fec:	c5 fd 59 c9          	vmulpd %ymm1,%ymm0,%ymm1
  400ff0:	c5 e5 5c c9          	vsubpd %ymm1,%ymm3,%ymm1
  400ff4:	c5 fd e6 c9          	vcvttpd2dq %ymm1,%xmm1
  400ff8:	c5 dc 54 e9          	vandps %ymm1,%ymm4,%ymm5
  400ffc:	c4 c3 7d 19 e8 01    	vextractf128 $0x1,%ymm5,%xmm8
  401002:	c5 f9 6f c5          	vmovdqa %xmm5,%xmm0
  401006:	c5 f9 76 c2          	vpcmpeqd %xmm2,%xmm0,%xmm0
  40100a:	c5 b9 76 d2          	vpcmpeqd %xmm2,%xmm8,%xmm2
  40100e:	c4 c3 7d 19 c8 01    	vextractf128 $0x1,%ymm1,%xmm8
  401014:	c4 e3 7d 18 c2 01    	vinsertf128 $0x1,%xmm2,%ymm0,%ymm0
  40101a:	c5 fc 55 02          	vandnps (%rdx),%ymm0,%ymm0
  40101e:	48 8d 15 cb 26 00 00 	lea    0x26cb(%rip),%rdx        # 4036f0 <_ZN2Vc2v03AVX9c_general9oneDoubleE>
  401025:	c5 fc 54 d4          	vandps %ymm4,%ymm0,%ymm2
  401029:	c4 e3 7d 19 d5 01    	vextractf128 $0x1,%ymm2,%xmm5
  40102f:	c5 f1 fe ca          	vpaddd %xmm2,%xmm1,%xmm1
  401033:	c5 f8 15 d0          	vunpckhps %xmm0,%xmm0,%xmm2
  401037:	c5 f8 14 c0          	vunpcklps %xmm0,%xmm0,%xmm0
  40103b:	c5 b9 fe ed          	vpaddd %xmm5,%xmm8,%xmm5
  40103f:	c4 e3 75 18 cd 01    	vinsertf128 $0x1,%xmm5,%ymm1,%ymm1
  401045:	c4 e2 7d 19 2a       	vbroadcastsd (%rdx),%ymm5
  40104a:	c4 e3 7d 18 c2 01    	vinsertf128 $0x1,%xmm2,%ymm0,%ymm0
  401050:	c4 e2 7d 19 50 08    	vbroadcastsd 0x8(%rax),%ymm2
  401056:	c5 fd 54 c5          	vandpd %ymm5,%ymm0,%ymm0
  40105a:	c5 e5 58 c0          	vaddpd %ymm0,%ymm3,%ymm0
  40105e:	c5 f4 54 1d fa 29 00 	vandps 0x29fa(%rip),%ymm1,%ymm3        # 403a60 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x20>
  401065:	00 
  401066:	c4 e2 7d 19 48 10    	vbroadcastsd 0x10(%rax),%ymm1
  40106c:	c5 fd 59 d2          	vmulpd %ymm2,%ymm0,%ymm2
  401070:	c5 fd 59 c9          	vmulpd %ymm1,%ymm0,%ymm1
  401074:	c5 cd 5c f2          	vsubpd %ymm2,%ymm6,%ymm6
  401078:	c5 cd 5c d1          	vsubpd %ymm1,%ymm6,%ymm2
  40107c:	c5 fd 6f 35 fc 29 00 	vmovdqa 0x29fc(%rip),%ymm6        # 403a80 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x40>
  401083:	00 
  401084:	c4 e2 7d 19 48 18    	vbroadcastsd 0x18(%rax),%ymm1
  40108a:	c4 e3 7d 19 f6 01    	vextractf128 $0x1,%ymm6,%xmm6
  401090:	c5 fd 59 c1          	vmulpd %ymm1,%ymm0,%ymm0
  401094:	c4 e3 7d 19 d9 01    	vextractf128 $0x1,%ymm3,%xmm1
  40109a:	c5 ed 5c d0          	vsubpd %ymm0,%ymm2,%ymm2
  40109e:	c5 e1 66 c6          	vpcmpgtd %xmm6,%xmm3,%xmm0
  4010a2:	c5 f1 66 f6          	vpcmpgtd %xmm6,%xmm1,%xmm6
  4010a6:	c4 e3 7d 18 f6 01    	vinsertf128 $0x1,%xmm6,%ymm0,%ymm6
  4010ac:	c5 f8 28 c6          	vmovaps %xmm6,%xmm0
  4010b0:	c5 cc 54 35 e8 29 00 	vandps 0x29e8(%rip),%ymm6,%ymm6        # 403aa0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x60>
  4010b7:	00 
  4010b8:	c5 78 15 c0          	vunpckhps %xmm0,%xmm0,%xmm8
  4010bc:	c5 f8 14 c0          	vunpcklps %xmm0,%xmm0,%xmm0
  4010c0:	c5 e1 fa de          	vpsubd %xmm6,%xmm3,%xmm3
  4010c4:	c4 c3 7d 18 c0 01    	vinsertf128 $0x1,%xmm8,%ymm0,%ymm0
  4010ca:	c4 c3 7d 19 f0 01    	vextractf128 $0x1,%ymm6,%xmm8
  4010d0:	c5 c4 57 c0          	vxorps %ymm0,%ymm7,%ymm0
  4010d4:	c5 f9 6f fb          	vmovdqa %xmm3,%xmm7
  4010d8:	c4 c1 71 fa c8       	vpsubd %xmm8,%xmm1,%xmm1
  4010dd:	c5 fd 6f 1d db 29 00 	vmovdqa 0x29db(%rip),%ymm3        # 403ac0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x80>
  4010e4:	00 
  4010e5:	c4 e3 45 18 f9 01    	vinsertf128 $0x1,%xmm1,%ymm7,%ymm7
  4010eb:	c4 e3 7d 19 db 01    	vextractf128 $0x1,%ymm3,%xmm3
  4010f1:	c5 ed 59 ca          	vmulpd %ymm2,%ymm2,%ymm1
  4010f5:	c4 e3 7d 19 fe 01    	vextractf128 $0x1,%ymm7,%xmm6
  4010fb:	c5 41 76 c3          	vpcmpeqd %xmm3,%xmm7,%xmm8
  4010ff:	c5 c1 76 fc          	vpcmpeqd %xmm4,%xmm7,%xmm7
  401103:	c5 c9 76 db          	vpcmpeqd %xmm3,%xmm6,%xmm3
  401107:	c4 63 3d 18 c3 01    	vinsertf128 $0x1,%xmm3,%ymm8,%ymm8
  40110d:	c4 e3 7d 19 e3 01    	vextractf128 $0x1,%ymm4,%xmm3
  401113:	c5 c9 76 f3          	vpcmpeqd %xmm3,%xmm6,%xmm6
  401117:	c5 f9 6f df          	vmovdqa %xmm7,%xmm3
  40111b:	c4 e2 7d 19 78 70    	vbroadcastsd 0x70(%rax),%ymm7
  401121:	c4 e3 65 18 de 01    	vinsertf128 $0x1,%xmm6,%ymm3,%ymm3
  401127:	c5 ed 59 f1          	vmulpd %ymm1,%ymm2,%ymm6
  40112b:	c4 c1 64 56 d8       	vorps  %ymm8,%ymm3,%ymm3
  401130:	c4 62 7d 19 40 78    	vbroadcastsd 0x78(%rax),%ymm8
  401136:	c5 e0 15 e3          	vunpckhps %xmm3,%xmm3,%xmm4
  40113a:	c5 e0 14 db          	vunpcklps %xmm3,%xmm3,%xmm3
  40113e:	c4 e3 65 18 dc 01    	vinsertf128 $0x1,%xmm4,%ymm3,%ymm3
  401144:	c4 e2 7d 19 a0 98 00 	vbroadcastsd 0x98(%rax),%ymm4
  40114b:	00 00 
  40114d:	c5 f5 59 e4          	vmulpd %ymm4,%ymm1,%ymm4
  401151:	c5 a5 58 e4          	vaddpd %ymm4,%ymm11,%ymm4
  401155:	c4 62 7d 19 58 60    	vbroadcastsd 0x60(%rax),%ymm11
  40115b:	c5 f5 59 e4          	vmulpd %ymm4,%ymm1,%ymm4
  40115f:	c5 ad 58 e4          	vaddpd %ymm4,%ymm10,%ymm4
  401163:	c4 62 7d 19 50 58    	vbroadcastsd 0x58(%rax),%ymm10
  401169:	c5 f5 59 e4          	vmulpd %ymm4,%ymm1,%ymm4
  40116d:	c5 b5 58 e4          	vaddpd %ymm4,%ymm9,%ymm4
  401171:	c4 62 7d 19 48 50    	vbroadcastsd 0x50(%rax),%ymm9
  401177:	c5 f5 59 e4          	vmulpd %ymm4,%ymm1,%ymm4
  40117b:	c5 bd 58 e4          	vaddpd %ymm4,%ymm8,%ymm4
  40117f:	c4 62 7d 19 40 48    	vbroadcastsd 0x48(%rax),%ymm8
  401185:	c5 f5 59 e4          	vmulpd %ymm4,%ymm1,%ymm4
  401189:	c5 c5 58 e4          	vaddpd %ymm4,%ymm7,%ymm4
  40118d:	c4 e2 7d 19 78 40    	vbroadcastsd 0x40(%rax),%ymm7
  401193:	c5 cd 59 e4          	vmulpd %ymm4,%ymm6,%ymm4
  401197:	c5 f5 59 f1          	vmulpd %ymm1,%ymm1,%ymm6
  40119b:	c5 ed 58 e4          	vaddpd %ymm4,%ymm2,%ymm4
  40119f:	c4 e2 7d 19 50 68    	vbroadcastsd 0x68(%rax),%ymm2
  4011a5:	48 8d 05 94 24 00 00 	lea    0x2494(%rip),%rax        # 403640 <_ZN2Vc2v03AVX5c_logIdE4dataE>
  4011ac:	c5 f5 59 d2          	vmulpd %ymm2,%ymm1,%ymm2
  4011b0:	c5 a5 58 d2          	vaddpd %ymm2,%ymm11,%ymm2
  4011b4:	c5 f5 59 d2          	vmulpd %ymm2,%ymm1,%ymm2
  4011b8:	c5 ad 58 d2          	vaddpd %ymm2,%ymm10,%ymm2
  4011bc:	c5 f5 59 d2          	vmulpd %ymm2,%ymm1,%ymm2
  4011c0:	c5 b5 58 d2          	vaddpd %ymm2,%ymm9,%ymm2
  4011c4:	c5 f5 59 d2          	vmulpd %ymm2,%ymm1,%ymm2
  4011c8:	c5 bd 58 d2          	vaddpd %ymm2,%ymm8,%ymm2
  4011cc:	c5 f5 59 d2          	vmulpd %ymm2,%ymm1,%ymm2
  4011d0:	c5 c5 58 d2          	vaddpd %ymm2,%ymm7,%ymm2
  4011d4:	c5 cd 59 d2          	vmulpd %ymm2,%ymm6,%ymm2
  4011d8:	c4 e2 7d 19 b0 90 00 	vbroadcastsd 0x90(%rax),%ymm6
  4011df:	00 00 
  4011e1:	c5 f5 59 ce          	vmulpd %ymm6,%ymm1,%ymm1
  4011e5:	c5 ed 5c c9          	vsubpd %ymm1,%ymm2,%ymm1
  4011e9:	c5 d5 58 e9          	vaddpd %ymm1,%ymm5,%ymm5
  4011ed:	c4 e3 5d 4b d5 30    	vblendvpd %ymm3,%ymm5,%ymm4,%ymm2
  4011f3:	c5 ed 57 0d e5 28 00 	vxorpd 0x28e5(%rip),%ymm2,%ymm1        # 403ae0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xa0>
  4011fa:	00 
  4011fb:	c4 e3 6d 4b c1 00    	vblendvpd %ymm0,%ymm1,%ymm2,%ymm0
  401201:	c3                   	retq   
  401202:	66 66 66 66 66 2e 0f 	data32 data32 data32 data32 nopw %cs:0x0(%rax,%rax,1)
  401209:	1f 84 00 00 00 00 00 

0000000000401210 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE3cosIdEENS0_3AVX6VectorIT_EERKSB_>:
  401210:	48 8d 05 05 25 00 00 	lea    0x2505(%rip),%rax        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  401217:	48 8d 15 22 28 00 00 	lea    0x2822(%rip),%rdx        # 403a40 <_ZN2Vc2v03AVX18_IndexesFromZero32E>
  40121e:	c5 e9 ef d2          	vpxor  %xmm2,%xmm2,%xmm2
  401222:	c4 e2 7d 19 28       	vbroadcastsd (%rax),%ymm5
  401227:	48 8d 05 12 26 00 00 	lea    0x2612(%rip),%rax        # 403840 <_ZN2Vc2v03AVX6c_trigIdE4dataE>
  40122e:	c5 d5 54 2f          	vandpd (%rdi),%ymm5,%ymm5
  401232:	c4 e2 7d 19 18       	vbroadcastsd (%rax),%ymm3
  401237:	c4 e2 7d 19 60 20    	vbroadcastsd 0x20(%rax),%ymm4
  40123d:	c4 e2 7d 19 40 28    	vbroadcastsd 0x28(%rax),%ymm0
  401243:	c4 62 7d 19 58 60    	vbroadcastsd 0x60(%rax),%ymm11
  401249:	c4 62 7d 19 50 58    	vbroadcastsd 0x58(%rax),%ymm10
  40124f:	c5 d5 5e db          	vdivpd %ymm3,%ymm5,%ymm3
  401253:	c4 e3 7d 09 db 03    	vroundpd $0x3,%ymm3,%ymm3
  401259:	c5 e5 59 e4          	vmulpd %ymm4,%ymm3,%ymm4
  40125d:	c4 e3 7d 09 e4 03    	vroundpd $0x3,%ymm4,%ymm4
  401263:	c5 fd 59 e4          	vmulpd %ymm4,%ymm0,%ymm4
  401267:	c4 e2 7d 18 42 04    	vbroadcastss 0x4(%rdx),%ymm0
  40126d:	48 8d 15 2c 23 00 00 	lea    0x232c(%rip),%rdx        # 4035a0 <_ZN2Vc2v06Common10AllBitsSetE>
  401274:	c5 e5 5c e4          	vsubpd %ymm4,%ymm3,%ymm4
  401278:	c5 fd e6 e4          	vcvttpd2dq %ymm4,%xmm4
  40127c:	c5 fc 54 f4          	vandps %ymm4,%ymm0,%ymm6
  401280:	c4 e3 7d 19 f7 01    	vextractf128 $0x1,%ymm6,%xmm7
  401286:	c5 f9 6f ce          	vmovdqa %xmm6,%xmm1
  40128a:	c5 f1 76 ca          	vpcmpeqd %xmm2,%xmm1,%xmm1
  40128e:	c5 c1 76 d2          	vpcmpeqd %xmm2,%xmm7,%xmm2
  401292:	c4 e3 7d 19 e7 01    	vextractf128 $0x1,%ymm4,%xmm7
  401298:	c4 e3 75 18 ca 01    	vinsertf128 $0x1,%xmm2,%ymm1,%ymm1
  40129e:	c5 f4 55 0a          	vandnps (%rdx),%ymm1,%ymm1
  4012a2:	48 8d 15 47 24 00 00 	lea    0x2447(%rip),%rdx        # 4036f0 <_ZN2Vc2v03AVX9c_general9oneDoubleE>
  4012a9:	c5 f4 54 d0          	vandps %ymm0,%ymm1,%ymm2
  4012ad:	c4 e3 7d 19 d6 01    	vextractf128 $0x1,%ymm2,%xmm6
  4012b3:	c5 d9 fe e2          	vpaddd %xmm2,%xmm4,%xmm4
  4012b7:	c5 f0 15 d1          	vunpckhps %xmm1,%xmm1,%xmm2
  4012bb:	c5 f0 14 c9          	vunpcklps %xmm1,%xmm1,%xmm1
  4012bf:	c5 c1 fe f6          	vpaddd %xmm6,%xmm7,%xmm6
  4012c3:	c4 e3 5d 18 e6 01    	vinsertf128 $0x1,%xmm6,%ymm4,%ymm4
  4012c9:	c5 fd 6f 3d af 27 00 	vmovdqa 0x27af(%rip),%ymm7        # 403a80 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x40>
  4012d0:	00 
  4012d1:	c4 e2 7d 19 32       	vbroadcastsd (%rdx),%ymm6
  4012d6:	48 8d 15 63 23 00 00 	lea    0x2363(%rip),%rdx        # 403640 <_ZN2Vc2v03AVX5c_logIdE4dataE>
  4012dd:	c4 e3 75 18 ca 01    	vinsertf128 $0x1,%xmm2,%ymm1,%ymm1
  4012e3:	c4 e2 7d 19 50 08    	vbroadcastsd 0x8(%rax),%ymm2
  4012e9:	c5 dc 54 25 6f 27 00 	vandps 0x276f(%rip),%ymm4,%ymm4        # 403a60 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x20>
  4012f0:	00 
  4012f1:	c5 f5 54 ce          	vandpd %ymm6,%ymm1,%ymm1
  4012f5:	c4 e3 7d 19 ff 01    	vextractf128 $0x1,%ymm7,%xmm7
  4012fb:	c5 e5 58 c9          	vaddpd %ymm1,%ymm3,%ymm1
  4012ff:	c4 e2 7d 19 58 10    	vbroadcastsd 0x10(%rax),%ymm3
  401305:	c5 f5 59 d2          	vmulpd %ymm2,%ymm1,%ymm2
  401309:	c5 f5 59 db          	vmulpd %ymm3,%ymm1,%ymm3
  40130d:	c5 d5 5c ea          	vsubpd %ymm2,%ymm5,%ymm5
  401311:	c5 d5 5c d3          	vsubpd %ymm3,%ymm5,%ymm2
  401315:	c4 e2 7d 19 58 18    	vbroadcastsd 0x18(%rax),%ymm3
  40131b:	c5 f5 59 cb          	vmulpd %ymm3,%ymm1,%ymm1
  40131f:	c4 e3 7d 19 e3 01    	vextractf128 $0x1,%ymm4,%xmm3
  401325:	c5 ed 5c d1          	vsubpd %ymm1,%ymm2,%ymm2
  401329:	c5 d9 66 cf          	vpcmpgtd %xmm7,%xmm4,%xmm1
  40132d:	c5 e1 66 ff          	vpcmpgtd %xmm7,%xmm3,%xmm7
  401331:	c4 e3 75 18 ff 01    	vinsertf128 $0x1,%xmm7,%ymm1,%ymm7
  401337:	c5 f8 28 cf          	vmovaps %xmm7,%xmm1
  40133b:	c5 c4 54 3d 5d 27 00 	vandps 0x275d(%rip),%ymm7,%ymm7        # 403aa0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x60>
  401342:	00 
  401343:	c5 f0 15 e9          	vunpckhps %xmm1,%xmm1,%xmm5
  401347:	c5 f0 14 c9          	vunpcklps %xmm1,%xmm1,%xmm1
  40134b:	c4 c3 7d 19 f8 01    	vextractf128 $0x1,%ymm7,%xmm8
  401351:	c5 d9 fa e7          	vpsubd %xmm7,%xmm4,%xmm4
  401355:	c4 e3 75 18 cd 01    	vinsertf128 $0x1,%xmm5,%ymm1,%ymm1
  40135b:	c5 f9 6f ec          	vmovdqa %xmm4,%xmm5
  40135f:	c4 e3 7d 19 c7 01    	vextractf128 $0x1,%ymm0,%xmm7
  401365:	c4 c1 61 fa d8       	vpsubd %xmm8,%xmm3,%xmm3
  40136a:	c4 e3 55 18 eb 01    	vinsertf128 $0x1,%xmm3,%ymm5,%ymm5
  401370:	c5 79 6f c0          	vmovdqa %xmm0,%xmm8
  401374:	c4 e3 7d 19 ec 01    	vextractf128 $0x1,%ymm5,%xmm4
  40137a:	c4 c1 51 66 c0       	vpcmpgtd %xmm8,%xmm5,%xmm0
  40137f:	c5 d9 66 df          	vpcmpgtd %xmm7,%xmm4,%xmm3
  401383:	c4 e3 7d 18 c3 01    	vinsertf128 $0x1,%xmm3,%ymm0,%ymm0
  401389:	c5 f8 15 d8          	vunpckhps %xmm0,%xmm0,%xmm3
  40138d:	c5 f8 14 c0          	vunpcklps %xmm0,%xmm0,%xmm0
  401391:	c4 e3 7d 18 c3 01    	vinsertf128 $0x1,%xmm3,%ymm0,%ymm0
  401397:	c5 fd 6f 1d 21 27 00 	vmovdqa 0x2721(%rip),%ymm3        # 403ac0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x80>
  40139e:	00 
  40139f:	c4 e3 7d 19 db 01    	vextractf128 $0x1,%ymm3,%xmm3
  4013a5:	c5 f4 57 c0          	vxorps %ymm0,%ymm1,%ymm0
  4013a9:	c5 ed 59 ca          	vmulpd %ymm2,%ymm2,%ymm1
  4013ad:	c5 51 76 cb          	vpcmpeqd %xmm3,%xmm5,%xmm9
  4013b1:	c5 d9 76 db          	vpcmpeqd %xmm3,%xmm4,%xmm3
  4013b5:	c5 d9 76 e7          	vpcmpeqd %xmm7,%xmm4,%xmm4
  4013b9:	c4 e2 7d 19 78 40    	vbroadcastsd 0x40(%rax),%ymm7
  4013bf:	c4 63 35 18 cb 01    	vinsertf128 $0x1,%xmm3,%ymm9,%ymm9
  4013c5:	c4 c1 51 76 d8       	vpcmpeqd %xmm8,%xmm5,%xmm3
  4013ca:	c4 62 7d 19 40 48    	vbroadcastsd 0x48(%rax),%ymm8
  4013d0:	c4 e3 65 18 dc 01    	vinsertf128 $0x1,%xmm4,%ymm3,%ymm3
  4013d6:	c5 f5 59 e9          	vmulpd %ymm1,%ymm1,%ymm5
  4013da:	c4 c1 64 56 d9       	vorps  %ymm9,%ymm3,%ymm3
  4013df:	c4 62 7d 19 48 50    	vbroadcastsd 0x50(%rax),%ymm9
  4013e5:	c5 e0 15 e3          	vunpckhps %xmm3,%xmm3,%xmm4
  4013e9:	c5 e0 14 db          	vunpcklps %xmm3,%xmm3,%xmm3
  4013ed:	c4 e3 65 18 dc 01    	vinsertf128 $0x1,%xmm4,%ymm3,%ymm3
  4013f3:	c4 e2 7d 19 60 68    	vbroadcastsd 0x68(%rax),%ymm4
  4013f9:	c5 f5 59 e4          	vmulpd %ymm4,%ymm1,%ymm4
  4013fd:	c5 a5 58 e4          	vaddpd %ymm4,%ymm11,%ymm4
  401401:	c4 62 7d 19 98 90 00 	vbroadcastsd 0x90(%rax),%ymm11
  401408:	00 00 
  40140a:	c5 f5 59 e4          	vmulpd %ymm4,%ymm1,%ymm4
  40140e:	c5 ad 58 e4          	vaddpd %ymm4,%ymm10,%ymm4
  401412:	c4 62 7d 19 90 88 00 	vbroadcastsd 0x88(%rax),%ymm10
  401419:	00 00 
  40141b:	c5 f5 59 e4          	vmulpd %ymm4,%ymm1,%ymm4
  40141f:	c5 b5 58 e4          	vaddpd %ymm4,%ymm9,%ymm4
  401423:	c4 62 7d 19 88 80 00 	vbroadcastsd 0x80(%rax),%ymm9
  40142a:	00 00 
  40142c:	c5 f5 59 e4          	vmulpd %ymm4,%ymm1,%ymm4
  401430:	c5 bd 58 e4          	vaddpd %ymm4,%ymm8,%ymm4
  401434:	c4 62 7d 19 40 78    	vbroadcastsd 0x78(%rax),%ymm8
  40143a:	c5 f5 59 e4          	vmulpd %ymm4,%ymm1,%ymm4
  40143e:	c5 c5 58 e4          	vaddpd %ymm4,%ymm7,%ymm4
  401442:	c4 e2 7d 19 b8 98 00 	vbroadcastsd 0x98(%rax),%ymm7
  401449:	00 00 
  40144b:	c5 f5 59 ff          	vmulpd %ymm7,%ymm1,%ymm7
  40144f:	c5 d5 59 e4          	vmulpd %ymm4,%ymm5,%ymm4
  401453:	c4 e2 7d 19 aa 90 00 	vbroadcastsd 0x90(%rdx),%ymm5
  40145a:	00 00 
  40145c:	c5 a5 58 ff          	vaddpd %ymm7,%ymm11,%ymm7
  401460:	c5 f5 59 ed          	vmulpd %ymm5,%ymm1,%ymm5
  401464:	c5 f5 59 ff          	vmulpd %ymm7,%ymm1,%ymm7
  401468:	c5 dd 5c e5          	vsubpd %ymm5,%ymm4,%ymm4
  40146c:	c4 e2 7d 19 68 70    	vbroadcastsd 0x70(%rax),%ymm5
  401472:	c5 ad 58 ff          	vaddpd %ymm7,%ymm10,%ymm7
  401476:	c5 cd 58 f4          	vaddpd %ymm4,%ymm6,%ymm6
  40147a:	c5 ed 59 e1          	vmulpd %ymm1,%ymm2,%ymm4
  40147e:	c5 f5 59 ff          	vmulpd %ymm7,%ymm1,%ymm7
  401482:	c5 b5 58 ff          	vaddpd %ymm7,%ymm9,%ymm7
  401486:	c5 f5 59 ff          	vmulpd %ymm7,%ymm1,%ymm7
  40148a:	c5 bd 58 ff          	vaddpd %ymm7,%ymm8,%ymm7
  40148e:	c5 f5 59 cf          	vmulpd %ymm7,%ymm1,%ymm1
  401492:	c5 d5 58 c9          	vaddpd %ymm1,%ymm5,%ymm1
  401496:	c5 dd 59 c9          	vmulpd %ymm1,%ymm4,%ymm1
  40149a:	c5 ed 58 d1          	vaddpd %ymm1,%ymm2,%ymm2
  40149e:	c4 e3 4d 4b d2 30    	vblendvpd %ymm3,%ymm2,%ymm6,%ymm2
  4014a4:	c5 ed 57 1d 34 26 00 	vxorpd 0x2634(%rip),%ymm2,%ymm3        # 403ae0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xa0>
  4014ab:	00 
  4014ac:	c4 e3 6d 4b c3 00    	vblendvpd %ymm0,%ymm3,%ymm2,%ymm0
  4014b2:	c3                   	retq   
  4014b3:	90                   	nop
  4014b4:	66 66 66 2e 0f 1f 84 	data32 data32 nopw %cs:0x0(%rax,%rax,1)
  4014bb:	00 00 00 00 00 

00000000004014c0 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE6sincosIdEEvRKNS0_3AVX6VectorIT_EEPSB_SE_>:
  4014c0:	48 8d 05 55 22 00 00 	lea    0x2255(%rip),%rax        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  4014c7:	48 8d 0d 72 25 00 00 	lea    0x2572(%rip),%rcx        # 403a40 <_ZN2Vc2v03AVX18_IndexesFromZero32E>
  4014ce:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
  4014d2:	4c 8d 05 c7 20 00 00 	lea    0x20c7(%rip),%r8        # 4035a0 <_ZN2Vc2v06Common10AllBitsSetE>
  4014d9:	c4 e2 7d 19 38       	vbroadcastsd (%rax),%ymm7
  4014de:	48 8d 05 5b 23 00 00 	lea    0x235b(%rip),%rax        # 403840 <_ZN2Vc2v03AVX6c_trigIdE4dataE>
  4014e5:	c4 e2 7d 18 69 04    	vbroadcastss 0x4(%rcx),%ymm5
  4014eb:	c5 c5 54 3f          	vandpd (%rdi),%ymm7,%ymm7
  4014ef:	c4 e2 7d 19 10       	vbroadcastsd (%rax),%ymm2
  4014f4:	c4 e2 7d 19 58 20    	vbroadcastsd 0x20(%rax),%ymm3
  4014fa:	c4 e2 7d 19 40 28    	vbroadcastsd 0x28(%rax),%ymm0
  401500:	c4 62 7d 19 60 60    	vbroadcastsd 0x60(%rax),%ymm12
  401506:	c4 62 7d 19 58 58    	vbroadcastsd 0x58(%rax),%ymm11
  40150c:	c4 62 7d 19 50 50    	vbroadcastsd 0x50(%rax),%ymm10
  401512:	c4 62 7d 19 48 48    	vbroadcastsd 0x48(%rax),%ymm9
  401518:	c5 c5 5e d2          	vdivpd %ymm2,%ymm7,%ymm2
  40151c:	c4 e3 7d 09 d2 03    	vroundpd $0x3,%ymm2,%ymm2
  401522:	c5 ed 59 db          	vmulpd %ymm3,%ymm2,%ymm3
  401526:	c4 e3 7d 09 db 03    	vroundpd $0x3,%ymm3,%ymm3
  40152c:	c5 fd 59 db          	vmulpd %ymm3,%ymm0,%ymm3
  401530:	c5 ed 5c db          	vsubpd %ymm3,%ymm2,%ymm3
  401534:	c5 fd e6 db          	vcvttpd2dq %ymm3,%xmm3
  401538:	c5 d4 54 e3          	vandps %ymm3,%ymm5,%ymm4
  40153c:	c4 c3 7d 19 d8 01    	vextractf128 $0x1,%ymm3,%xmm8
  401542:	c4 e3 7d 19 e6 01    	vextractf128 $0x1,%ymm4,%xmm6
  401548:	c5 f9 6f c4          	vmovdqa %xmm4,%xmm0
  40154c:	c5 f9 76 c1          	vpcmpeqd %xmm1,%xmm0,%xmm0
  401550:	c5 c9 76 c9          	vpcmpeqd %xmm1,%xmm6,%xmm1
  401554:	c4 e3 7d 18 c1 01    	vinsertf128 $0x1,%xmm1,%ymm0,%ymm0
  40155a:	c4 c1 7c 55 00       	vandnps (%r8),%ymm0,%ymm0
  40155f:	4c 8d 05 8a 21 00 00 	lea    0x218a(%rip),%r8        # 4036f0 <_ZN2Vc2v03AVX9c_general9oneDoubleE>
  401566:	c5 fc 54 cd          	vandps %ymm5,%ymm0,%ymm1
  40156a:	c5 e1 fe f1          	vpaddd %xmm1,%xmm3,%xmm6
  40156e:	c4 e3 7d 19 cc 01    	vextractf128 $0x1,%ymm1,%xmm4
  401574:	c5 f8 15 c8          	vunpckhps %xmm0,%xmm0,%xmm1
  401578:	c5 f8 14 c0          	vunpcklps %xmm0,%xmm0,%xmm0
  40157c:	c5 b9 fe e4          	vpaddd %xmm4,%xmm8,%xmm4
  401580:	c4 e3 4d 18 f4 01    	vinsertf128 $0x1,%xmm4,%ymm6,%ymm6
  401586:	c4 c2 7d 19 20       	vbroadcastsd (%r8),%ymm4
  40158b:	4c 8d 05 ae 20 00 00 	lea    0x20ae(%rip),%r8        # 403640 <_ZN2Vc2v03AVX5c_logIdE4dataE>
  401592:	c4 e3 7d 18 c1 01    	vinsertf128 $0x1,%xmm1,%ymm0,%ymm0
  401598:	c4 e2 7d 19 48 08    	vbroadcastsd 0x8(%rax),%ymm1
  40159e:	c5 cc 54 35 ba 24 00 	vandps 0x24ba(%rip),%ymm6,%ymm6        # 403a60 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x20>
  4015a5:	00 
  4015a6:	c5 fd 54 c4          	vandpd %ymm4,%ymm0,%ymm0
  4015aa:	c5 ed 58 c0          	vaddpd %ymm0,%ymm2,%ymm0
  4015ae:	c4 e2 7d 19 50 10    	vbroadcastsd 0x10(%rax),%ymm2
  4015b4:	c5 fd 59 c9          	vmulpd %ymm1,%ymm0,%ymm1
  4015b8:	c5 fd 59 d2          	vmulpd %ymm2,%ymm0,%ymm2
  4015bc:	c5 c5 5c f9          	vsubpd %ymm1,%ymm7,%ymm7
  4015c0:	c5 c5 5c ca          	vsubpd %ymm2,%ymm7,%ymm1
  4015c4:	c4 e2 7d 19 50 18    	vbroadcastsd 0x18(%rax),%ymm2
  4015ca:	c5 fd 59 c2          	vmulpd %ymm2,%ymm0,%ymm0
  4015ce:	c4 e3 7d 19 f2 01    	vextractf128 $0x1,%ymm6,%xmm2
  4015d4:	c5 f5 5c c8          	vsubpd %ymm0,%ymm1,%ymm1
  4015d8:	c5 fd 6f 05 a0 24 00 	vmovdqa 0x24a0(%rip),%ymm0        # 403a80 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x40>
  4015df:	00 
  4015e0:	c4 e3 7d 19 c0 01    	vextractf128 $0x1,%ymm0,%xmm0
  4015e6:	c5 c9 66 d8          	vpcmpgtd %xmm0,%xmm6,%xmm3
  4015ea:	c5 e9 66 c0          	vpcmpgtd %xmm0,%xmm2,%xmm0
  4015ee:	c4 e3 65 18 d8 01    	vinsertf128 $0x1,%xmm0,%ymm3,%ymm3
  4015f4:	c5 f8 28 c3          	vmovaps %xmm3,%xmm0
  4015f8:	c5 78 15 c0          	vunpckhps %xmm0,%xmm0,%xmm8
  4015fc:	c5 f8 14 c0          	vunpcklps %xmm0,%xmm0,%xmm0
  401600:	c5 f8 28 f8          	vmovaps %xmm0,%xmm7
  401604:	c5 e4 54 05 94 24 00 	vandps 0x2494(%rip),%ymm3,%ymm0        # 403aa0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x60>
  40160b:	00 
  40160c:	c4 c3 45 18 f8 01    	vinsertf128 $0x1,%xmm8,%ymm7,%ymm7
  401612:	c5 c9 fa f0          	vpsubd %xmm0,%xmm6,%xmm6
  401616:	c4 c3 7d 19 c0 01    	vextractf128 $0x1,%ymm0,%xmm8
  40161c:	c5 f5 59 c1          	vmulpd %ymm1,%ymm1,%ymm0
  401620:	c5 f9 6f de          	vmovdqa %xmm6,%xmm3
  401624:	c4 c1 69 fa d0       	vpsubd %xmm8,%xmm2,%xmm2
  401629:	c4 62 7d 19 40 68    	vbroadcastsd 0x68(%rax),%ymm8
  40162f:	c4 e2 7d 19 70 40    	vbroadcastsd 0x40(%rax),%ymm6
  401635:	c4 e3 65 18 da 01    	vinsertf128 $0x1,%xmm2,%ymm3,%ymm3
  40163b:	c4 41 7d 59 c0       	vmulpd %ymm8,%ymm0,%ymm8
  401640:	c5 fd 59 d0          	vmulpd %ymm0,%ymm0,%ymm2
  401644:	c4 41 1d 58 c0       	vaddpd %ymm8,%ymm12,%ymm8
  401649:	c4 62 7d 19 a0 90 00 	vbroadcastsd 0x90(%rax),%ymm12
  401650:	00 00 
  401652:	c4 41 7d 59 c0       	vmulpd %ymm8,%ymm0,%ymm8
  401657:	c4 41 25 58 c0       	vaddpd %ymm8,%ymm11,%ymm8
  40165c:	c4 62 7d 19 98 88 00 	vbroadcastsd 0x88(%rax),%ymm11
  401663:	00 00 
  401665:	c4 41 7d 59 c0       	vmulpd %ymm8,%ymm0,%ymm8
  40166a:	c4 41 2d 58 c0       	vaddpd %ymm8,%ymm10,%ymm8
  40166f:	c4 62 7d 19 90 80 00 	vbroadcastsd 0x80(%rax),%ymm10
  401676:	00 00 
  401678:	c4 41 7d 59 c0       	vmulpd %ymm8,%ymm0,%ymm8
  40167d:	c4 41 35 58 c0       	vaddpd %ymm8,%ymm9,%ymm8
  401682:	c4 62 7d 19 48 78    	vbroadcastsd 0x78(%rax),%ymm9
  401688:	c4 41 7d 59 c0       	vmulpd %ymm8,%ymm0,%ymm8
  40168d:	c4 41 4d 58 c0       	vaddpd %ymm8,%ymm6,%ymm8
  401692:	c4 e2 7d 19 b0 98 00 	vbroadcastsd 0x98(%rax),%ymm6
  401699:	00 00 
  40169b:	c5 fd 59 f6          	vmulpd %ymm6,%ymm0,%ymm6
  40169f:	c4 41 6d 59 c0       	vmulpd %ymm8,%ymm2,%ymm8
  4016a4:	c4 c2 7d 19 90 90 00 	vbroadcastsd 0x90(%r8),%ymm2
  4016ab:	00 00 
  4016ad:	c5 9d 58 f6          	vaddpd %ymm6,%ymm12,%ymm6
  4016b1:	c5 fd 59 d2          	vmulpd %ymm2,%ymm0,%ymm2
  4016b5:	c5 fd 59 f6          	vmulpd %ymm6,%ymm0,%ymm6
  4016b9:	c5 3d 5c c2          	vsubpd %ymm2,%ymm8,%ymm8
  4016bd:	c5 f5 59 d0          	vmulpd %ymm0,%ymm1,%ymm2
  4016c1:	c5 a5 58 f6          	vaddpd %ymm6,%ymm11,%ymm6
  4016c5:	c4 41 5d 58 c0       	vaddpd %ymm8,%ymm4,%ymm8
  4016ca:	c4 e2 7d 19 60 70    	vbroadcastsd 0x70(%rax),%ymm4
  4016d0:	c5 fd 59 f6          	vmulpd %ymm6,%ymm0,%ymm6
  4016d4:	c5 ad 58 f6          	vaddpd %ymm6,%ymm10,%ymm6
  4016d8:	c5 fd 59 f6          	vmulpd %ymm6,%ymm0,%ymm6
  4016dc:	c5 b5 58 f6          	vaddpd %ymm6,%ymm9,%ymm6
  4016e0:	c5 7d 6f 0d d8 23 00 	vmovdqa 0x23d8(%rip),%ymm9        # 403ac0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x80>
  4016e7:	00 
  4016e8:	c4 43 7d 19 c9 01    	vextractf128 $0x1,%ymm9,%xmm9
  4016ee:	c5 fd 59 c6          	vmulpd %ymm6,%ymm0,%ymm0
  4016f2:	c4 e3 7d 19 ee 01    	vextractf128 $0x1,%ymm5,%xmm6
  4016f8:	c5 dd 58 c0          	vaddpd %ymm0,%ymm4,%ymm0
  4016fc:	c5 f9 6f e5          	vmovdqa %xmm5,%xmm4
  401700:	c5 e1 76 ec          	vpcmpeqd %xmm4,%xmm3,%xmm5
  401704:	c5 e1 66 e4          	vpcmpgtd %xmm4,%xmm3,%xmm4
  401708:	c5 ed 59 c0          	vmulpd %ymm0,%ymm2,%ymm0
  40170c:	c4 e3 7d 19 da 01    	vextractf128 $0x1,%ymm3,%xmm2
  401712:	c5 f5 58 c0          	vaddpd %ymm0,%ymm1,%ymm0
  401716:	c4 c1 61 76 c9       	vpcmpeqd %xmm9,%xmm3,%xmm1
  40171b:	c4 41 69 76 c9       	vpcmpeqd %xmm9,%xmm2,%xmm9
  401720:	c4 43 75 18 c9 01    	vinsertf128 $0x1,%xmm9,%ymm1,%ymm9
  401726:	c5 e9 76 ce          	vpcmpeqd %xmm6,%xmm2,%xmm1
  40172a:	c5 e9 66 f6          	vpcmpgtd %xmm6,%xmm2,%xmm6
  40172e:	c4 e3 55 18 e9 01    	vinsertf128 $0x1,%xmm1,%ymm5,%ymm5
  401734:	c4 e3 5d 18 e6 01    	vinsertf128 $0x1,%xmm6,%ymm4,%ymm4
  40173a:	c5 fd 28 35 9e 23 00 	vmovapd 0x239e(%rip),%ymm6        # 403ae0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xa0>
  401741:	00 
  401742:	c4 c1 54 56 e9       	vorps  %ymm9,%ymm5,%ymm5
  401747:	c5 50 15 d5          	vunpckhps %xmm5,%xmm5,%xmm10
  40174b:	c5 d0 14 ed          	vunpcklps %xmm5,%xmm5,%xmm5
  40174f:	c5 f8 28 cd          	vmovaps %xmm5,%xmm1
  401753:	c5 d8 15 ec          	vunpckhps %xmm4,%xmm4,%xmm5
  401757:	c5 d8 14 e4          	vunpcklps %xmm4,%xmm4,%xmm4
  40175b:	c4 c3 75 18 ca 01    	vinsertf128 $0x1,%xmm10,%ymm1,%ymm1
  401761:	c4 e3 5d 18 e5 01    	vinsertf128 $0x1,%xmm5,%ymm4,%ymm4
  401767:	c4 e3 3d 4b c8 10    	vblendvpd %ymm1,%ymm0,%ymm8,%ymm1
  40176d:	c5 c4 57 e4          	vxorps %ymm4,%ymm7,%ymm4
  401771:	c5 f5 57 ee          	vxorpd %ymm6,%ymm1,%ymm5
  401775:	c4 e3 75 4b cd 40    	vblendvpd %ymm4,%ymm5,%ymm1,%ymm1
  40177b:	c5 fd 29 0a          	vmovapd %ymm1,(%rdx)
  40177f:	c4 e2 7d 18 61 04    	vbroadcastss 0x4(%rcx),%ymm4
  401785:	c4 e3 7d 19 e1 01    	vextractf128 $0x1,%ymm4,%xmm1
  40178b:	c5 e1 76 dc          	vpcmpeqd %xmm4,%xmm3,%xmm3
  40178f:	c5 e9 76 d1          	vpcmpeqd %xmm1,%xmm2,%xmm2
  401793:	c5 f9 6f cb          	vmovdqa %xmm3,%xmm1
  401797:	c4 e3 75 18 ca 01    	vinsertf128 $0x1,%xmm2,%ymm1,%ymm1
  40179d:	c4 c1 74 56 c9       	vorps  %ymm9,%ymm1,%ymm1
  4017a2:	c5 f0 15 d1          	vunpckhps %xmm1,%xmm1,%xmm2
  4017a6:	c5 f0 14 c9          	vunpcklps %xmm1,%xmm1,%xmm1
  4017aa:	c4 e3 75 18 ca 01    	vinsertf128 $0x1,%xmm2,%ymm1,%ymm1
  4017b0:	c5 e9 57 d2          	vxorpd %xmm2,%xmm2,%xmm2
  4017b4:	c4 c3 7d 4b c8 10    	vblendvpd %ymm1,%ymm8,%ymm0,%ymm1
  4017ba:	c5 fd 28 07          	vmovapd (%rdi),%ymm0
  4017be:	c5 fd c2 c2 01       	vcmpltpd %ymm2,%ymm0,%ymm0
  4017c3:	c5 f5 57 f6          	vxorpd %ymm6,%ymm1,%ymm6
  4017c7:	c5 c4 57 f8          	vxorps %ymm0,%ymm7,%ymm7
  4017cb:	c4 e3 75 4b ce 70    	vblendvpd %ymm7,%ymm6,%ymm1,%ymm1
  4017d1:	c5 fd 29 0e          	vmovapd %ymm1,(%rsi)
  4017d5:	c5 f8 77             	vzeroupper 
  4017d8:	c3                   	retq   
  4017d9:	90                   	nop
  4017da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004017e0 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE4asinIdEENS0_3AVX6VectorIT_EERKSB_>:
  4017e0:	48 8d 05 35 1f 00 00 	lea    0x1f35(%rip),%rax        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  4017e7:	c5 fd 28 1f          	vmovapd (%rdi),%ymm3
  4017eb:	48 8d 15 ae 1d 00 00 	lea    0x1dae(%rip),%rdx        # 4035a0 <_ZN2Vc2v06Common10AllBitsSetE>
  4017f2:	c5 c1 57 ff          	vxorpd %xmm7,%xmm7,%xmm7
  4017f6:	c4 e2 7d 19 10       	vbroadcastsd (%rax),%ymm2
  4017fb:	48 8d 05 ee 1e 00 00 	lea    0x1eee(%rip),%rax        # 4036f0 <_ZN2Vc2v03AVX9c_general9oneDoubleE>
  401802:	c5 e5 c2 ff 01       	vcmpltpd %ymm7,%ymm3,%ymm7
  401807:	c5 e5 54 d2          	vandpd %ymm2,%ymm3,%ymm2
  40180b:	c4 e2 7d 19 08       	vbroadcastsd (%rax),%ymm1
  401810:	48 8d 05 29 20 00 00 	lea    0x2029(%rip),%rax        # 403840 <_ZN2Vc2v03AVX6c_trigIdE4dataE>
  401817:	c5 ed c2 f1 06       	vcmpnlepd %ymm1,%ymm2,%ymm6
  40181c:	c4 62 7d 19 90 40 01 	vbroadcastsd 0x140(%rax),%ymm10
  401823:	00 00 
  401825:	c5 f5 5c ca          	vsubpd %ymm2,%ymm1,%ymm1
  401829:	c4 62 7d 19 a8 48 01 	vbroadcastsd 0x148(%rax),%ymm13
  401830:	00 00 
  401832:	c4 62 7d 19 a0 50 01 	vbroadcastsd 0x150(%rax),%ymm12
  401839:	00 00 
  40183b:	c4 62 7d 19 98 58 01 	vbroadcastsd 0x158(%rax),%ymm11
  401842:	00 00 
  401844:	c4 62 7d 19 88 60 01 	vbroadcastsd 0x160(%rax),%ymm9
  40184b:	00 00 
  40184d:	c4 62 7d 19 b8 90 01 	vbroadcastsd 0x190(%rax),%ymm15
  401854:	00 00 
  401856:	c4 41 75 59 d2       	vmulpd %ymm10,%ymm1,%ymm10
  40185b:	c5 f5 58 e9          	vaddpd %ymm1,%ymm1,%ymm5
  40185f:	c4 62 7d 19 b0 98 01 	vbroadcastsd 0x198(%rax),%ymm14
  401866:	00 00 
  401868:	c4 e2 7d 19 20       	vbroadcastsd (%rax),%ymm4
  40186d:	c4 62 7d 19 80 30 01 	vbroadcastsd 0x130(%rax),%ymm8
  401874:	00 00 
  401876:	c4 41 15 58 d2       	vaddpd %ymm10,%ymm13,%ymm10
  40187b:	c4 62 7d 19 a8 70 01 	vbroadcastsd 0x170(%rax),%ymm13
  401882:	00 00 
  401884:	c5 fd 51 ed          	vsqrtpd %ymm5,%ymm5
  401888:	c4 e2 7d 19 80 28 01 	vbroadcastsd 0x128(%rax),%ymm0
  40188f:	00 00 
  401891:	c4 41 6d c2 c0 06    	vcmpnlepd %ymm8,%ymm2,%ymm8
  401897:	c5 ed c2 c0 01       	vcmpltpd %ymm0,%ymm2,%ymm0
  40189c:	c4 41 75 59 d2       	vmulpd %ymm10,%ymm1,%ymm10
  4018a1:	c5 3c 55 02          	vandnps (%rdx),%ymm8,%ymm8
  4018a5:	c4 41 1d 58 d2       	vaddpd %ymm10,%ymm12,%ymm10
  4018aa:	c4 62 7d 19 a0 78 01 	vbroadcastsd 0x178(%rax),%ymm12
  4018b1:	00 00 
  4018b3:	c4 41 75 59 d2       	vmulpd %ymm10,%ymm1,%ymm10
  4018b8:	c4 41 25 58 d2       	vaddpd %ymm10,%ymm11,%ymm10
  4018bd:	c4 62 7d 19 98 68 01 	vbroadcastsd 0x168(%rax),%ymm11
  4018c4:	00 00 
  4018c6:	c4 41 75 58 db       	vaddpd %ymm11,%ymm1,%ymm11
  4018cb:	c4 41 75 59 d2       	vmulpd %ymm10,%ymm1,%ymm10
  4018d0:	c4 41 75 59 db       	vmulpd %ymm11,%ymm1,%ymm11
  4018d5:	c4 41 35 58 d2       	vaddpd %ymm10,%ymm9,%ymm10
  4018da:	c4 62 7d 19 88 80 01 	vbroadcastsd 0x180(%rax),%ymm9
  4018e1:	00 00 
  4018e3:	c4 41 15 58 db       	vaddpd %ymm11,%ymm13,%ymm11
  4018e8:	c4 62 7d 19 a8 a0 01 	vbroadcastsd 0x1a0(%rax),%ymm13
  4018ef:	00 00 
  4018f1:	c4 41 75 59 d2       	vmulpd %ymm10,%ymm1,%ymm10
  4018f6:	c4 41 75 59 db       	vmulpd %ymm11,%ymm1,%ymm11
  4018fb:	c4 41 1d 58 db       	vaddpd %ymm11,%ymm12,%ymm11
  401900:	c4 62 7d 19 a0 a8 01 	vbroadcastsd 0x1a8(%rax),%ymm12
  401907:	00 00 
  401909:	c4 c1 75 59 cb       	vmulpd %ymm11,%ymm1,%ymm1
  40190e:	c4 62 7d 19 98 b0 01 	vbroadcastsd 0x1b0(%rax),%ymm11
  401915:	00 00 
  401917:	c5 b5 58 c9          	vaddpd %ymm1,%ymm9,%ymm1
  40191b:	c4 62 7d 19 88 88 01 	vbroadcastsd 0x188(%rax),%ymm9
  401922:	00 00 
  401924:	c5 2d 5e d1          	vdivpd %ymm1,%ymm10,%ymm10
  401928:	c5 ed 59 ca          	vmulpd %ymm2,%ymm2,%ymm1
  40192c:	c4 41 75 59 c9       	vmulpd %ymm9,%ymm1,%ymm9
  401931:	c4 41 05 58 c9       	vaddpd %ymm9,%ymm15,%ymm9
  401936:	c4 62 7d 19 b8 c0 01 	vbroadcastsd 0x1c0(%rax),%ymm15
  40193d:	00 00 
  40193f:	c4 41 75 59 c9       	vmulpd %ymm9,%ymm1,%ymm9
  401944:	c4 41 0d 58 c9       	vaddpd %ymm9,%ymm14,%ymm9
  401949:	c4 62 7d 19 b0 c8 01 	vbroadcastsd 0x1c8(%rax),%ymm14
  401950:	00 00 
  401952:	c4 41 75 59 c9       	vmulpd %ymm9,%ymm1,%ymm9
  401957:	c4 41 15 58 c9       	vaddpd %ymm9,%ymm13,%ymm9
  40195c:	c4 62 7d 19 a8 d0 01 	vbroadcastsd 0x1d0(%rax),%ymm13
  401963:	00 00 
  401965:	c4 41 75 59 c9       	vmulpd %ymm9,%ymm1,%ymm9
  40196a:	c4 41 1d 58 c9       	vaddpd %ymm9,%ymm12,%ymm9
  40196f:	c4 62 7d 19 a0 b8 01 	vbroadcastsd 0x1b8(%rax),%ymm12
  401976:	00 00 
  401978:	c4 41 75 58 e4       	vaddpd %ymm12,%ymm1,%ymm12
  40197d:	c4 41 75 59 c9       	vmulpd %ymm9,%ymm1,%ymm9
  401982:	c4 41 75 59 e4       	vmulpd %ymm12,%ymm1,%ymm12
  401987:	c4 41 25 58 c9       	vaddpd %ymm9,%ymm11,%ymm9
  40198c:	c4 62 7d 19 98 d8 01 	vbroadcastsd 0x1d8(%rax),%ymm11
  401993:	00 00 
  401995:	c4 41 05 58 e4       	vaddpd %ymm12,%ymm15,%ymm12
  40199a:	c4 41 75 59 c9       	vmulpd %ymm9,%ymm1,%ymm9
  40199f:	c4 41 75 59 e4       	vmulpd %ymm12,%ymm1,%ymm12
  4019a4:	c4 41 0d 58 e4       	vaddpd %ymm12,%ymm14,%ymm12
  4019a9:	c4 41 75 59 e4       	vmulpd %ymm12,%ymm1,%ymm12
  4019ae:	c4 41 15 58 e4       	vaddpd %ymm12,%ymm13,%ymm12
  4019b3:	c4 c1 75 59 cc       	vmulpd %ymm12,%ymm1,%ymm1
  4019b8:	c5 a5 58 c9          	vaddpd %ymm1,%ymm11,%ymm1
  4019bc:	c5 b5 5e c9          	vdivpd %ymm1,%ymm9,%ymm1
  4019c0:	c5 5d 5c cd          	vsubpd %ymm5,%ymm4,%ymm9
  4019c4:	c4 c1 55 59 ea       	vmulpd %ymm10,%ymm5,%ymm5
  4019c9:	c4 62 7d 19 90 20 01 	vbroadcastsd 0x120(%rax),%ymm10
  4019d0:	00 00 
  4019d2:	c4 c1 55 5c ea       	vsubpd %ymm10,%ymm5,%ymm5
  4019d7:	c5 ed 59 c9          	vmulpd %ymm1,%ymm2,%ymm1
  4019db:	c5 b5 5c ed          	vsubpd %ymm5,%ymm9,%ymm5
  4019df:	c5 ed 58 d1          	vaddpd %ymm1,%ymm2,%ymm2
  4019e3:	c5 dd 58 e5          	vaddpd %ymm5,%ymm4,%ymm4
  4019e7:	c4 e3 5d 4b d2 80    	vblendvpd %ymm8,%ymm2,%ymm4,%ymm2
  4019ed:	c5 ed 57 0d eb 20 00 	vxorpd 0x20eb(%rip),%ymm2,%ymm1        # 403ae0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xa0>
  4019f4:	00 
  4019f5:	c4 e3 6d 4b d1 70    	vblendvpd %ymm7,%ymm1,%ymm2,%ymm2
  4019fb:	c4 e3 6d 4b db 00    	vblendvpd %ymm0,%ymm3,%ymm2,%ymm3
  401a01:	c5 e5 56 c6          	vorpd  %ymm6,%ymm3,%ymm0
  401a05:	c3                   	retq   
  401a06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401a0d:	00 00 00 

0000000000401a10 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE4atanIdEENS0_3AVX6VectorIT_EERKSB_>:
  401a10:	48 8d 05 05 1d 00 00 	lea    0x1d05(%rip),%rax        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  401a17:	c5 fd 28 17          	vmovapd (%rdi),%ymm2
  401a1b:	48 8d 15 ce 1c 00 00 	lea    0x1cce(%rip),%rdx        # 4036f0 <_ZN2Vc2v03AVX9c_general9oneDoubleE>
  401a22:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
  401a26:	c4 e2 7d 19 00       	vbroadcastsd (%rax),%ymm0
  401a2b:	48 8d 05 0e 1e 00 00 	lea    0x1e0e(%rip),%rax        # 403840 <_ZN2Vc2v03AVX6c_trigIdE4dataE>
  401a32:	c4 e2 7d 19 2a       	vbroadcastsd (%rdx),%ymm5
  401a37:	c5 ed 54 c0          	vandpd %ymm0,%ymm2,%ymm0
  401a3b:	c5 6d c2 c1 01       	vcmpltpd %ymm1,%ymm2,%ymm8
  401a40:	c4 e2 7d 19 a0 10 01 	vbroadcastsd 0x110(%rax),%ymm4
  401a47:	00 00 
  401a49:	c4 e2 7d 19 98 18 01 	vbroadcastsd 0x118(%rax),%ymm3
  401a50:	00 00 
  401a52:	c5 fd 58 f5          	vaddpd %ymm5,%ymm0,%ymm6
  401a56:	c4 62 7d 19 10       	vbroadcastsd (%rax),%ymm10
  401a5b:	c5 6d 59 c9          	vmulpd %ymm1,%ymm2,%ymm9
  401a5f:	c5 fd 5c fd          	vsubpd %ymm5,%ymm0,%ymm7
  401a63:	c4 62 7d 19 b0 c8 00 	vbroadcastsd 0xc8(%rax),%ymm14
  401a6a:	00 00 
  401a6c:	c5 fd c2 e4 06       	vcmpnlepd %ymm4,%ymm0,%ymm4
  401a71:	c4 62 7d 19 a8 d0 00 	vbroadcastsd 0xd0(%rax),%ymm13
  401a78:	00 00 
  401a7a:	c4 62 7d 19 a0 d8 00 	vbroadcastsd 0xd8(%rax),%ymm12
  401a81:	00 00 
  401a83:	c5 fd c2 db 06       	vcmpnlepd %ymm3,%ymm0,%ymm3
  401a88:	c4 62 7d 19 b8 f0 00 	vbroadcastsd 0xf0(%rax),%ymm15
  401a8f:	00 00 
  401a91:	c4 62 7d 19 98 e0 00 	vbroadcastsd 0xe0(%rax),%ymm11
  401a98:	00 00 
  401a9a:	c4 41 6d c2 c9 07    	vcmpordpd %ymm9,%ymm2,%ymm9
  401aa0:	c4 c3 75 4b ca 30    	vblendvpd %ymm3,%ymm10,%ymm1,%ymm1
  401aa6:	c5 ed c2 d2 03       	vcmpunordpd %ymm2,%ymm2,%ymm2
  401aab:	c5 c5 5e fe          	vdivpd %ymm6,%ymm7,%ymm7
  401aaf:	c5 fd 28 35 29 20 00 	vmovapd 0x2029(%rip),%ymm6        # 403ae0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xa0>
  401ab6:	00 
  401ab7:	c5 d5 57 ee          	vxorpd %ymm6,%ymm5,%ymm5
  401abb:	c5 d5 5e e8          	vdivpd %ymm0,%ymm5,%ymm5
  401abf:	c4 e3 45 4b ed 40    	vblendvpd %ymm4,%ymm5,%ymm7,%ymm5
  401ac5:	c4 e2 7d 19 b8 c0 00 	vbroadcastsd 0xc0(%rax),%ymm7
  401acc:	00 00 
  401ace:	c4 e3 7d 4b c5 30    	vblendvpd %ymm3,%ymm5,%ymm0,%ymm0
  401ad4:	c4 e2 7d 19 a8 b0 00 	vbroadcastsd 0xb0(%rax),%ymm5
  401adb:	00 00 
  401add:	c4 63 75 4b d5 40    	vblendvpd %ymm4,%ymm5,%ymm1,%ymm10
  401ae3:	c5 fd 59 c8          	vmulpd %ymm0,%ymm0,%ymm1
  401ae7:	c5 f5 59 ff          	vmulpd %ymm7,%ymm1,%ymm7
  401aeb:	c5 8d 58 ff          	vaddpd %ymm7,%ymm14,%ymm7
  401aef:	c4 62 7d 19 b0 f8 00 	vbroadcastsd 0xf8(%rax),%ymm14
  401af6:	00 00 
  401af8:	c5 f5 59 ff          	vmulpd %ymm7,%ymm1,%ymm7
  401afc:	c5 95 58 ff          	vaddpd %ymm7,%ymm13,%ymm7
  401b00:	c4 62 7d 19 a8 00 01 	vbroadcastsd 0x100(%rax),%ymm13
  401b07:	00 00 
  401b09:	c5 f5 59 ff          	vmulpd %ymm7,%ymm1,%ymm7
  401b0d:	c5 9d 58 ff          	vaddpd %ymm7,%ymm12,%ymm7
  401b11:	c4 62 7d 19 a0 e8 00 	vbroadcastsd 0xe8(%rax),%ymm12
  401b18:	00 00 
  401b1a:	c4 41 75 58 e4       	vaddpd %ymm12,%ymm1,%ymm12
  401b1f:	c5 f5 59 ff          	vmulpd %ymm7,%ymm1,%ymm7
  401b23:	c4 41 75 59 e4       	vmulpd %ymm12,%ymm1,%ymm12
  401b28:	c5 a5 58 ff          	vaddpd %ymm7,%ymm11,%ymm7
  401b2c:	c4 62 7d 19 98 08 01 	vbroadcastsd 0x108(%rax),%ymm11
  401b33:	00 00 
  401b35:	c4 41 05 58 e4       	vaddpd %ymm12,%ymm15,%ymm12
  401b3a:	c5 f5 59 ff          	vmulpd %ymm7,%ymm1,%ymm7
  401b3e:	c4 41 75 59 e4       	vmulpd %ymm12,%ymm1,%ymm12
  401b43:	c4 41 0d 58 e4       	vaddpd %ymm12,%ymm14,%ymm12
  401b48:	c4 41 75 59 e4       	vmulpd %ymm12,%ymm1,%ymm12
  401b4d:	c4 41 15 58 e4       	vaddpd %ymm12,%ymm13,%ymm12
  401b52:	c4 c1 75 59 cc       	vmulpd %ymm12,%ymm1,%ymm1
  401b57:	c5 a5 58 c9          	vaddpd %ymm1,%ymm11,%ymm1
  401b5b:	c5 c5 5e c9          	vdivpd %ymm1,%ymm7,%ymm1
  401b5f:	c4 e2 7d 19 b8 20 01 	vbroadcastsd 0x120(%rax),%ymm7
  401b66:	00 00 
  401b68:	48 8d 05 31 1a 00 00 	lea    0x1a31(%rip),%rax        # 4035a0 <_ZN2Vc2v06Common10AllBitsSetE>
  401b6f:	c5 dc 55 20          	vandnps (%rax),%ymm4,%ymm4
  401b73:	48 8d 05 c6 1a 00 00 	lea    0x1ac6(%rip),%rax        # 403640 <_ZN2Vc2v03AVX5c_logIdE4dataE>
  401b7a:	c4 62 7d 19 98 90 00 	vbroadcastsd 0x90(%rax),%ymm11
  401b81:	00 00 
  401b83:	c5 fd 59 c9          	vmulpd %ymm1,%ymm0,%ymm1
  401b87:	c4 41 45 59 db       	vmulpd %ymm11,%ymm7,%ymm11
  401b8c:	c5 fd 58 c1          	vaddpd %ymm1,%ymm0,%ymm0
  401b90:	c4 c3 45 4b fb 40    	vblendvpd %ymm4,%ymm11,%ymm7,%ymm7
  401b96:	c5 e5 54 df          	vandpd %ymm7,%ymm3,%ymm3
  401b9a:	c5 fd 58 c3          	vaddpd %ymm3,%ymm0,%ymm0
  401b9e:	c5 ad 58 c0          	vaddpd %ymm0,%ymm10,%ymm0
  401ba2:	c4 e3 55 4b c0 90    	vblendvpd %ymm9,%ymm0,%ymm5,%ymm0
  401ba8:	c5 fd 57 f6          	vxorpd %ymm6,%ymm0,%ymm6
  401bac:	c4 e3 7d 4b c6 80    	vblendvpd %ymm8,%ymm6,%ymm0,%ymm0
  401bb2:	c5 fd 56 c2          	vorpd  %ymm2,%ymm0,%ymm0
  401bb6:	c3                   	retq   
  401bb7:	90                   	nop
  401bb8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401bbf:	00 

0000000000401bc0 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE5atan2IdEENS0_3AVX6VectorIT_EERKSB_SD_>:
  401bc0:	55                   	push   %rbp
  401bc1:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
  401bc5:	48 89 e5             	mov    %rsp,%rbp
  401bc8:	41 57                	push   %r15
  401bca:	49 89 ff             	mov    %rdi,%r15
  401bcd:	41 56                	push   %r14
  401bcf:	41 55                	push   %r13
  401bd1:	41 54                	push   %r12
  401bd3:	49 89 f4             	mov    %rsi,%r12
  401bd6:	53                   	push   %rbx
  401bd7:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  401bdb:	48 81 ec 40 01 00 00 	sub    $0x140,%rsp
  401be2:	4c 8d 2d 2b 1b 00 00 	lea    0x1b2b(%rip),%r13        # 403714 <_ZN2Vc2v03AVX9c_general13signMaskFloatE>
  401be9:	48 8d 05 b0 19 00 00 	lea    0x19b0(%rip),%rax        # 4035a0 <_ZN2Vc2v06Common10AllBitsSetE>
  401bf0:	c5 fd 28 07          	vmovapd (%rdi),%ymm0
  401bf4:	48 8d 1d 45 1c 00 00 	lea    0x1c45(%rip),%rbx        # 403840 <_ZN2Vc2v03AVX6c_trigIdE4dataE>
  401bfb:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  401c02:	00 
  401c03:	c5 fd 28 16          	vmovapd (%rsi),%ymm2
  401c07:	4c 8d 35 0e 1b 00 00 	lea    0x1b0e(%rip),%r14        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  401c0e:	c5 fd c2 e9 00       	vcmpeqpd %ymm1,%ymm0,%ymm5
  401c13:	c4 62 7d 19 8b b8 00 	vbroadcastsd 0xb8(%rbx),%ymm9
  401c1a:	00 00 
  401c1c:	c5 ed c2 e1 00       	vcmpeqpd %ymm1,%ymm2,%ymm4
  401c21:	c5 fd 29 ac 24 00 01 	vmovapd %ymm5,0x100(%rsp)
  401c28:	00 00 
  401c2a:	c4 42 7d 19 45 00    	vbroadcastsd 0x0(%r13),%ymm8
  401c30:	c5 fd 29 24 24       	vmovapd %ymm4,(%rsp)
  401c35:	c5 bd 54 ea          	vandpd %ymm2,%ymm8,%ymm5
  401c39:	c4 41 7d 54 c0       	vandpd %ymm8,%ymm0,%ymm8
  401c3e:	c4 e3 7d 19 ee 01    	vextractf128 $0x1,%ymm5,%xmm6
  401c44:	c5 f9 6f dd          	vmovdqa %xmm5,%xmm3
  401c48:	c5 c9 72 e6 1f       	vpsrad $0x1f,%xmm6,%xmm6
  401c4d:	c5 e1 72 e3 1f       	vpsrad $0x1f,%xmm3,%xmm3
  401c52:	c4 e3 65 18 de 01    	vinsertf128 $0x1,%xmm6,%ymm3,%ymm3
  401c58:	c5 ed 59 f1          	vmulpd %ymm1,%ymm2,%ymm6
  401c5c:	c4 e3 7d 04 db f5    	vpermilps $0xf5,%ymm3,%ymm3
  401c62:	c5 ed c2 f6 07       	vcmpordpd %ymm6,%ymm2,%ymm6
  401c67:	c5 e4 54 fc          	vandps %ymm4,%ymm3,%ymm7
  401c6b:	c5 fc 28 18          	vmovaps (%rax),%ymm3
  401c6f:	48 8d 05 7a 1a 00 00 	lea    0x1a7a(%rip),%rax        # 4036f0 <_ZN2Vc2v03AVX9c_general9oneDoubleE>
  401c76:	c5 fd c2 e1 01       	vcmpltpd %ymm1,%ymm0,%ymm4
  401c7b:	c5 cc 55 f3          	vandnps %ymm3,%ymm6,%ymm6
  401c7f:	c5 fc 29 5c 24 20    	vmovaps %ymm3,0x20(%rsp)
  401c85:	c5 fc 29 bc 24 e0 00 	vmovaps %ymm7,0xe0(%rsp)
  401c8c:	00 00 
  401c8e:	c5 fd 29 a4 24 c0 00 	vmovapd %ymm4,0xc0(%rsp)
  401c95:	00 00 
  401c97:	c5 fc 29 b4 24 a0 00 	vmovaps %ymm6,0xa0(%rsp)
  401c9e:	00 00 
  401ca0:	c5 fd 59 f1          	vmulpd %ymm1,%ymm0,%ymm6
  401ca4:	c4 c2 7d 19 3e       	vbroadcastsd (%r14),%ymm7
  401ca9:	c5 fd 29 4c 24 40    	vmovapd %ymm1,0x40(%rsp)
  401caf:	c5 35 54 cf          	vandpd %ymm7,%ymm9,%ymm9
  401cb3:	c5 fd c2 f6 07       	vcmpordpd %ymm6,%ymm0,%ymm6
  401cb8:	c4 41 3d 56 c1       	vorpd  %ymm9,%ymm8,%ymm8
  401cbd:	c5 6d c2 c9 05       	vcmpnltpd %ymm1,%ymm2,%ymm9
  401cc2:	c5 cc 55 f3          	vandnps %ymm3,%ymm6,%ymm6
  401cc6:	c4 c1 35 55 c8       	vandnpd %ymm8,%ymm9,%ymm1
  401ccb:	c5 fd 29 8c 24 80 00 	vmovapd %ymm1,0x80(%rsp)
  401cd2:	00 00 
  401cd4:	c4 62 7d 19 00       	vbroadcastsd (%rax),%ymm8
  401cd9:	c5 fc 29 74 24 60    	vmovaps %ymm6,0x60(%rsp)
  401cdf:	c5 bd 54 ff          	vandpd %ymm7,%ymm8,%ymm7
  401ce3:	c5 d5 56 ef          	vorpd  %ymm7,%ymm5,%ymm5
  401ce7:	c4 e3 6d 4b d5 60    	vblendvpd %ymm6,%ymm5,%ymm2,%ymm2
  401ced:	c5 fd 5e c2          	vdivpd %ymm2,%ymm0,%ymm0
  401cf1:	c5 fd 29 84 24 20 01 	vmovapd %ymm0,0x120(%rsp)
  401cf8:	00 00 
  401cfa:	c5 f8 77             	vzeroupper 
  401cfd:	e8 0e fd ff ff       	callq  401a10 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE4atanIdEENS0_3AVX6VectorIT_EERKSB_>
  401d02:	c5 fd 28 24 24       	vmovapd (%rsp),%ymm4
  401d07:	c5 fd 58 84 24 80 00 	vaddpd 0x80(%rsp),%ymm0,%ymm0
  401d0e:	00 00 
  401d10:	c5 5c 54 84 24 00 01 	vandps 0x100(%rsp),%ymm4,%ymm8
  401d17:	00 00 
  401d19:	c4 c1 7d 28 17       	vmovapd (%r15),%ymm2
  401d1e:	c4 c2 7d 19 2e       	vbroadcastsd (%r14),%ymm5
  401d23:	c4 c2 7d 19 7d 00    	vbroadcastsd 0x0(%r13),%ymm7
  401d29:	c4 62 7d 19 8b b8 00 	vbroadcastsd 0xb8(%rbx),%ymm9
  401d30:	00 00 
  401d32:	c5 fc 28 5c 24 20    	vmovaps 0x20(%rsp),%ymm3
  401d38:	c5 3d 55 c0          	vandnpd %ymm0,%ymm8,%ymm8
  401d3c:	c5 fc 28 74 24 60    	vmovaps 0x60(%rsp),%ymm6
  401d42:	c5 ed 54 c7          	vandpd %ymm7,%ymm2,%ymm0
  401d46:	c5 fd 28 4c 24 40    	vmovapd 0x40(%rsp),%ymm1
  401d4c:	c5 35 54 cd          	vandpd %ymm5,%ymm9,%ymm9
  401d50:	c5 ed 55 db          	vandnpd %ymm3,%ymm2,%ymm3
  401d54:	c5 ed c2 d2 03       	vcmpunordpd %ymm2,%ymm2,%ymm2
  401d59:	c5 dc 54 a4 24 c0 00 	vandps 0xc0(%rsp),%ymm4,%ymm4
  401d60:	00 00 
  401d62:	c4 41 7d 56 c9       	vorpd  %ymm9,%ymm0,%ymm9
  401d67:	c4 c1 65 57 1c 24    	vxorpd (%r12),%ymm3,%ymm3
  401d6d:	c5 35 54 8c 24 e0 00 	vandpd 0xe0(%rsp),%ymm9,%ymm9
  401d74:	00 00 
  401d76:	c5 cc 54 b4 24 a0 00 	vandps 0xa0(%rsp),%ymm6,%ymm6
  401d7d:	00 00 
  401d7f:	c5 e5 54 ff          	vandpd %ymm7,%ymm3,%ymm7
  401d83:	c4 c1 7d 28 1c 24    	vmovapd (%r12),%ymm3
  401d89:	c4 41 3d 58 c1       	vaddpd %ymm9,%ymm8,%ymm8
  401d8e:	c4 62 7d 19 8b b0 00 	vbroadcastsd 0xb0(%rbx),%ymm9
  401d95:	00 00 
  401d97:	c5 e5 c2 db 03       	vcmpunordpd %ymm3,%ymm3,%ymm3
  401d9c:	c5 35 57 0d 3c 1d 00 	vxorpd 0x1d3c(%rip),%ymm9,%ymm9        # 403ae0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xa0>
  401da3:	00 
  401da4:	c5 ec 56 d3          	vorps  %ymm3,%ymm2,%ymm2
  401da8:	c4 c3 3d 4b e1 40    	vblendvpd %ymm4,%ymm9,%ymm8,%ymm4
  401dae:	c4 62 7d 19 03       	vbroadcastsd (%rbx),%ymm8
  401db3:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  401db7:	c5 3d 54 c5          	vandpd %ymm5,%ymm8,%ymm8
  401dbb:	5b                   	pop    %rbx
  401dbc:	41 5c                	pop    %r12
  401dbe:	c4 c1 45 56 f8       	vorpd  %ymm8,%ymm7,%ymm7
  401dc3:	41 5d                	pop    %r13
  401dc5:	41 5e                	pop    %r14
  401dc7:	c5 cd 54 f7          	vandpd %ymm7,%ymm6,%ymm6
  401dcb:	41 5f                	pop    %r15
  401dcd:	5d                   	pop    %rbp
  401dce:	c5 dd 58 e6          	vaddpd %ymm6,%ymm4,%ymm4
  401dd2:	c5 dd 54 ed          	vandpd %ymm5,%ymm4,%ymm5
  401dd6:	c5 dd c2 c9 00       	vcmpeqpd %ymm1,%ymm4,%ymm1
  401ddb:	c5 fd 56 c5          	vorpd  %ymm5,%ymm0,%ymm0
  401ddf:	c4 e3 5d 4b e0 10    	vblendvpd %ymm1,%ymm0,%ymm4,%ymm4
  401de5:	c5 dd 56 c2          	vorpd  %ymm2,%ymm4,%ymm0
  401de9:	c3                   	retq   
  401dea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401df0 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE3sinIfEENS0_3AVX6VectorIT_EERKSB_>:
  401df0:	48 8d 05 25 19 00 00 	lea    0x1925(%rip),%rax        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  401df7:	c5 fc 28 3f          	vmovaps (%rdi),%ymm7
  401dfb:	48 8d 15 3e 1c 00 00 	lea    0x1c3e(%rip),%rdx        # 403a40 <_ZN2Vc2v03AVX18_IndexesFromZero32E>
  401e02:	c4 e2 7d 18 48 04    	vbroadcastss 0x4(%rax),%ymm1
  401e08:	48 8d 05 31 19 00 00 	lea    0x1931(%rip),%rax        # 403740 <_ZN2Vc2v03AVX6c_trigIfE4dataE>
  401e0f:	c4 e2 7d 18 5a 04    	vbroadcastss 0x4(%rdx),%ymm3
  401e15:	48 8d 15 f0 18 00 00 	lea    0x18f0(%rip),%rdx        # 40370c <_ZN2Vc2v03AVX9c_general8oneFloatE>
  401e1c:	c5 c4 54 c9          	vandps %ymm1,%ymm7,%ymm1
  401e20:	c4 e2 7d 18 40 54    	vbroadcastss 0x54(%rax),%ymm0
  401e26:	c5 f4 59 c0          	vmulps %ymm0,%ymm1,%ymm0
  401e2a:	c5 fe 5b c0          	vcvttps2dq %ymm0,%ymm0
  401e2e:	c5 e4 54 d0          	vandps %ymm0,%ymm3,%ymm2
  401e32:	c4 e3 7d 19 c6 01    	vextractf128 $0x1,%ymm0,%xmm6
  401e38:	c5 f9 6f e0          	vmovdqa %xmm0,%xmm4
  401e3c:	c4 e3 7d 19 d5 01    	vextractf128 $0x1,%ymm2,%xmm5
  401e42:	c5 d9 fe e2          	vpaddd %xmm2,%xmm4,%xmm4
  401e46:	c4 e2 7d 18 50 04    	vbroadcastss 0x4(%rax),%ymm2
  401e4c:	c5 c9 fe ed          	vpaddd %xmm5,%xmm6,%xmm5
  401e50:	c4 e3 5d 18 e5 01    	vinsertf128 $0x1,%xmm5,%ymm4,%ymm4
  401e56:	c4 e2 7d 18 68 08    	vbroadcastss 0x8(%rax),%ymm5
  401e5c:	c5 fd 6f 35 1c 1c 00 	vmovdqa 0x1c1c(%rip),%ymm6        # 403a80 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x40>
  401e63:	00 
  401e64:	c5 fc 5b c4          	vcvtdq2ps %ymm4,%ymm0
  401e68:	c5 fc 59 d2          	vmulps %ymm2,%ymm0,%ymm2
  401e6c:	c5 fc 59 ed          	vmulps %ymm5,%ymm0,%ymm5
  401e70:	c5 dc 54 25 e8 1b 00 	vandps 0x1be8(%rip),%ymm4,%ymm4        # 403a60 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x20>
  401e77:	00 
  401e78:	c5 f4 5c ca          	vsubps %ymm2,%ymm1,%ymm1
  401e7c:	c4 e3 7d 19 f6 01    	vextractf128 $0x1,%ymm6,%xmm6
  401e82:	c5 f4 5c d5          	vsubps %ymm5,%ymm1,%ymm2
  401e86:	c4 e2 7d 18 48 0c    	vbroadcastss 0xc(%rax),%ymm1
  401e8c:	c5 fc 59 c1          	vmulps %ymm1,%ymm0,%ymm0
  401e90:	c4 e3 7d 19 e1 01    	vextractf128 $0x1,%ymm4,%xmm1
  401e96:	c5 ec 5c d0          	vsubps %ymm0,%ymm2,%ymm2
  401e9a:	c5 d9 66 c6          	vpcmpgtd %xmm6,%xmm4,%xmm0
  401e9e:	c5 f1 66 f6          	vpcmpgtd %xmm6,%xmm1,%xmm6
  401ea2:	c4 e3 7d 18 f6 01    	vinsertf128 $0x1,%xmm6,%ymm0,%ymm6
  401ea8:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  401eac:	c5 c4 c2 c0 01       	vcmpltps %ymm0,%ymm7,%ymm0
  401eb1:	c5 fc 57 c6          	vxorps %ymm6,%ymm0,%ymm0
  401eb5:	c5 cc 54 35 e3 1b 00 	vandps 0x1be3(%rip),%ymm6,%ymm6        # 403aa0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x60>
  401ebc:	00 
  401ebd:	c4 e3 7d 19 f7 01    	vextractf128 $0x1,%ymm6,%xmm7
  401ec3:	c5 d9 fa e6          	vpsubd %xmm6,%xmm4,%xmm4
  401ec7:	c5 fd 6f 35 f1 1b 00 	vmovdqa 0x1bf1(%rip),%ymm6        # 403ac0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x80>
  401ece:	00 
  401ecf:	c5 f9 6f ec          	vmovdqa %xmm4,%xmm5
  401ed3:	c4 e3 7d 19 f6 01    	vextractf128 $0x1,%ymm6,%xmm6
  401ed9:	c5 f1 fa cf          	vpsubd %xmm7,%xmm1,%xmm1
  401edd:	c4 e3 55 18 e9 01    	vinsertf128 $0x1,%xmm1,%ymm5,%ymm5
  401ee3:	c4 e2 7d 18 3a       	vbroadcastss (%rdx),%ymm7
  401ee8:	c5 ec 59 ca          	vmulps %ymm2,%ymm2,%ymm1
  401eec:	c4 e3 7d 19 ec 01    	vextractf128 $0x1,%ymm5,%xmm4
  401ef2:	c5 51 76 c6          	vpcmpeqd %xmm6,%xmm5,%xmm8
  401ef6:	c5 d1 76 eb          	vpcmpeqd %xmm3,%xmm5,%xmm5
  401efa:	c5 d9 76 f6          	vpcmpeqd %xmm6,%xmm4,%xmm6
  401efe:	c4 e3 3d 18 f6 01    	vinsertf128 $0x1,%xmm6,%ymm8,%ymm6
  401f04:	c4 c3 7d 19 d8 01    	vextractf128 $0x1,%ymm3,%xmm8
  401f0a:	c4 e2 7d 18 58 40    	vbroadcastss 0x40(%rax),%ymm3
  401f10:	c5 f4 59 db          	vmulps %ymm3,%ymm1,%ymm3
  401f14:	c4 c1 59 76 e0       	vpcmpeqd %xmm8,%xmm4,%xmm4
  401f19:	c4 62 7d 18 40 3c    	vbroadcastss 0x3c(%rax),%ymm8
  401f1f:	c4 e3 55 18 e4 01    	vinsertf128 $0x1,%xmm4,%ymm5,%ymm4
  401f25:	c5 ec 59 e9          	vmulps %ymm1,%ymm2,%ymm5
  401f29:	c5 bc 58 db          	vaddps %ymm3,%ymm8,%ymm3
  401f2d:	c4 62 7d 18 40 24    	vbroadcastss 0x24(%rax),%ymm8
  401f33:	c5 dc 56 e6          	vorps  %ymm6,%ymm4,%ymm4
  401f37:	c4 e2 7d 18 70 38    	vbroadcastss 0x38(%rax),%ymm6
  401f3d:	c5 f4 59 db          	vmulps %ymm3,%ymm1,%ymm3
  401f41:	c5 cc 58 db          	vaddps %ymm3,%ymm6,%ymm3
  401f45:	c4 e2 7d 18 70 20    	vbroadcastss 0x20(%rax),%ymm6
  401f4b:	c5 d4 59 db          	vmulps %ymm3,%ymm5,%ymm3
  401f4f:	c5 f4 59 e9          	vmulps %ymm1,%ymm1,%ymm5
  401f53:	c5 ec 58 db          	vaddps %ymm3,%ymm2,%ymm3
  401f57:	c4 e2 7d 18 50 28    	vbroadcastss 0x28(%rax),%ymm2
  401f5d:	48 8d 05 5c 16 00 00 	lea    0x165c(%rip),%rax        # 4035c0 <_ZN2Vc2v03AVX5c_logIfE4dataE>
  401f64:	c5 f4 59 d2          	vmulps %ymm2,%ymm1,%ymm2
  401f68:	c5 bc 58 d2          	vaddps %ymm2,%ymm8,%ymm2
  401f6c:	c5 f4 59 d2          	vmulps %ymm2,%ymm1,%ymm2
  401f70:	c5 cc 58 d2          	vaddps %ymm2,%ymm6,%ymm2
  401f74:	c5 d4 59 d2          	vmulps %ymm2,%ymm5,%ymm2
  401f78:	c4 e2 7d 18 68 48    	vbroadcastss 0x48(%rax),%ymm5
  401f7e:	c5 f4 59 cd          	vmulps %ymm5,%ymm1,%ymm1
  401f82:	c5 ec 5c c9          	vsubps %ymm1,%ymm2,%ymm1
  401f86:	c5 c4 58 c9          	vaddps %ymm1,%ymm7,%ymm1
  401f8a:	c4 e3 65 4a d1 40    	vblendvps %ymm4,%ymm1,%ymm3,%ymm2
  401f90:	c5 ec 57 0d 68 1b 00 	vxorps 0x1b68(%rip),%ymm2,%ymm1        # 403b00 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xc0>
  401f97:	00 
  401f98:	c4 e3 6d 4a c1 00    	vblendvps %ymm0,%ymm1,%ymm2,%ymm0
  401f9e:	c3                   	retq   
  401f9f:	90                   	nop

0000000000401fa0 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE3cosIfEENS0_3AVX6VectorIT_EERKSB_>:
  401fa0:	48 8d 05 75 17 00 00 	lea    0x1775(%rip),%rax        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  401fa7:	48 8d 15 92 1a 00 00 	lea    0x1a92(%rip),%rdx        # 403a40 <_ZN2Vc2v03AVX18_IndexesFromZero32E>
  401fae:	c4 e2 7d 18 58 04    	vbroadcastss 0x4(%rax),%ymm3
  401fb4:	48 8d 05 85 17 00 00 	lea    0x1785(%rip),%rax        # 403740 <_ZN2Vc2v03AVX6c_trigIfE4dataE>
  401fbb:	c4 e2 7d 18 42 04    	vbroadcastss 0x4(%rdx),%ymm0
  401fc1:	48 8d 15 44 17 00 00 	lea    0x1744(%rip),%rdx        # 40370c <_ZN2Vc2v03AVX9c_general8oneFloatE>
  401fc8:	c5 e4 54 1f          	vandps (%rdi),%ymm3,%ymm3
  401fcc:	c4 e2 7d 18 48 54    	vbroadcastss 0x54(%rax),%ymm1
  401fd2:	c4 62 7d 18 0a       	vbroadcastss (%rdx),%ymm9
  401fd7:	48 8d 15 e2 15 00 00 	lea    0x15e2(%rip),%rdx        # 4035c0 <_ZN2Vc2v03AVX5c_logIfE4dataE>
  401fde:	c5 e4 59 c9          	vmulps %ymm1,%ymm3,%ymm1
  401fe2:	c5 fe 5b c9          	vcvttps2dq %ymm1,%ymm1
  401fe6:	c5 fc 54 d1          	vandps %ymm1,%ymm0,%ymm2
  401fea:	c4 e3 7d 19 ce 01    	vextractf128 $0x1,%ymm1,%xmm6
  401ff0:	c5 f9 6f e9          	vmovdqa %xmm1,%xmm5
  401ff4:	c4 e3 7d 19 d4 01    	vextractf128 $0x1,%ymm2,%xmm4
  401ffa:	c5 d1 fe ea          	vpaddd %xmm2,%xmm5,%xmm5
  401ffe:	c4 e2 7d 18 50 04    	vbroadcastss 0x4(%rax),%ymm2
  402004:	c5 c9 fe e4          	vpaddd %xmm4,%xmm6,%xmm4
  402008:	c4 e3 55 18 ec 01    	vinsertf128 $0x1,%xmm4,%ymm5,%ymm5
  40200e:	c4 e2 7d 18 60 08    	vbroadcastss 0x8(%rax),%ymm4
  402014:	c5 fc 5b cd          	vcvtdq2ps %ymm5,%ymm1
  402018:	c5 f4 59 d2          	vmulps %ymm2,%ymm1,%ymm2
  40201c:	c5 f4 59 e4          	vmulps %ymm4,%ymm1,%ymm4
  402020:	c5 d4 54 2d 38 1a 00 	vandps 0x1a38(%rip),%ymm5,%ymm5        # 403a60 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x20>
  402027:	00 
  402028:	c5 e4 5c da          	vsubps %ymm2,%ymm3,%ymm3
  40202c:	c5 e4 5c d4          	vsubps %ymm4,%ymm3,%ymm2
  402030:	c4 e2 7d 18 58 0c    	vbroadcastss 0xc(%rax),%ymm3
  402036:	c5 f4 59 cb          	vmulps %ymm3,%ymm1,%ymm1
  40203a:	c4 e3 7d 19 eb 01    	vextractf128 $0x1,%ymm5,%xmm3
  402040:	c5 ec 5c d1          	vsubps %ymm1,%ymm2,%ymm2
  402044:	c5 fd 6f 0d 34 1a 00 	vmovdqa 0x1a34(%rip),%ymm1        # 403a80 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x40>
  40204b:	00 
  40204c:	c4 e3 7d 19 c9 01    	vextractf128 $0x1,%ymm1,%xmm1
  402052:	c5 d1 66 e1          	vpcmpgtd %xmm1,%xmm5,%xmm4
  402056:	c5 e1 66 c9          	vpcmpgtd %xmm1,%xmm3,%xmm1
  40205a:	c4 e3 5d 18 c9 01    	vinsertf128 $0x1,%xmm1,%ymm4,%ymm1
  402060:	c5 f4 54 35 38 1a 00 	vandps 0x1a38(%rip),%ymm1,%ymm6        # 403aa0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x60>
  402067:	00 
  402068:	c4 e3 7d 19 f7 01    	vextractf128 $0x1,%ymm6,%xmm7
  40206e:	c5 d1 fa ee          	vpsubd %xmm6,%xmm5,%xmm5
  402072:	c5 f9 6f e5          	vmovdqa %xmm5,%xmm4
  402076:	c4 e3 7d 19 c5 01    	vextractf128 $0x1,%ymm0,%xmm5
  40207c:	c5 e1 fa df          	vpsubd %xmm7,%xmm3,%xmm3
  402080:	c4 e3 5d 18 e3 01    	vinsertf128 $0x1,%xmm3,%ymm4,%ymm4
  402086:	c4 e3 7d 19 e3 01    	vextractf128 $0x1,%ymm4,%xmm3
  40208c:	c5 d9 66 f8          	vpcmpgtd %xmm0,%xmm4,%xmm7
  402090:	c5 d9 76 c0          	vpcmpeqd %xmm0,%xmm4,%xmm0
  402094:	c5 e1 66 f5          	vpcmpgtd %xmm5,%xmm3,%xmm6
  402098:	c5 e1 76 ed          	vpcmpeqd %xmm5,%xmm3,%xmm5
  40209c:	c4 e3 45 18 fe 01    	vinsertf128 $0x1,%xmm6,%ymm7,%ymm7
  4020a2:	c5 fd 6f 35 16 1a 00 	vmovdqa 0x1a16(%rip),%ymm6        # 403ac0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x80>
  4020a9:	00 
  4020aa:	c4 e3 7d 19 f6 01    	vextractf128 $0x1,%ymm6,%xmm6
  4020b0:	c5 f4 57 ff          	vxorps %ymm7,%ymm1,%ymm7
  4020b4:	c5 ec 59 ca          	vmulps %ymm2,%ymm2,%ymm1
  4020b8:	c5 59 76 c6          	vpcmpeqd %xmm6,%xmm4,%xmm8
  4020bc:	c4 e2 7d 18 60 20    	vbroadcastss 0x20(%rax),%ymm4
  4020c2:	c5 e1 76 f6          	vpcmpeqd %xmm6,%xmm3,%xmm6
  4020c6:	c5 f9 6f d8          	vmovdqa %xmm0,%xmm3
  4020ca:	c4 e3 3d 18 f6 01    	vinsertf128 $0x1,%xmm6,%ymm8,%ymm6
  4020d0:	c4 62 7d 18 40 28    	vbroadcastss 0x28(%rax),%ymm8
  4020d6:	c4 e3 65 18 dd 01    	vinsertf128 $0x1,%xmm5,%ymm3,%ymm3
  4020dc:	c4 e2 7d 18 68 24    	vbroadcastss 0x24(%rax),%ymm5
  4020e2:	c4 41 74 59 c0       	vmulps %ymm8,%ymm1,%ymm8
  4020e7:	c5 f4 59 c1          	vmulps %ymm1,%ymm1,%ymm0
  4020eb:	c5 e4 56 de          	vorps  %ymm6,%ymm3,%ymm3
  4020ef:	c4 e2 7d 18 70 3c    	vbroadcastss 0x3c(%rax),%ymm6
  4020f5:	c4 41 54 58 c0       	vaddps %ymm8,%ymm5,%ymm8
  4020fa:	c4 e2 7d 18 68 40    	vbroadcastss 0x40(%rax),%ymm5
  402100:	c5 f4 59 ed          	vmulps %ymm5,%ymm1,%ymm5
  402104:	c4 41 74 59 c0       	vmulps %ymm8,%ymm1,%ymm8
  402109:	c5 cc 58 ed          	vaddps %ymm5,%ymm6,%ymm5
  40210d:	c4 41 5c 58 c0       	vaddps %ymm8,%ymm4,%ymm8
  402112:	c4 e2 7d 18 60 38    	vbroadcastss 0x38(%rax),%ymm4
  402118:	c4 41 7c 59 c0       	vmulps %ymm8,%ymm0,%ymm8
  40211d:	c4 e2 7d 18 42 48    	vbroadcastss 0x48(%rdx),%ymm0
  402123:	c5 f4 59 c0          	vmulps %ymm0,%ymm1,%ymm0
  402127:	c5 3c 5c c0          	vsubps %ymm0,%ymm8,%ymm8
  40212b:	c5 ec 59 c1          	vmulps %ymm1,%ymm2,%ymm0
  40212f:	c5 f4 59 cd          	vmulps %ymm5,%ymm1,%ymm1
  402133:	c4 41 34 58 c0       	vaddps %ymm8,%ymm9,%ymm8
  402138:	c5 dc 58 c9          	vaddps %ymm1,%ymm4,%ymm1
  40213c:	c5 fc 59 c9          	vmulps %ymm1,%ymm0,%ymm1
  402140:	c5 ec 58 d1          	vaddps %ymm1,%ymm2,%ymm2
  402144:	c4 e3 3d 4a d2 30    	vblendvps %ymm3,%ymm2,%ymm8,%ymm2
  40214a:	c5 ec 57 1d ae 19 00 	vxorps 0x19ae(%rip),%ymm2,%ymm3        # 403b00 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xc0>
  402151:	00 
  402152:	c4 e3 6d 4a c3 70    	vblendvps %ymm7,%ymm3,%ymm2,%ymm0
  402158:	c3                   	retq   
  402159:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000402160 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE6sincosIfEEvRKNS0_3AVX6VectorIT_EEPSB_SE_>:
  402160:	48 8d 05 b5 15 00 00 	lea    0x15b5(%rip),%rax        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  402167:	48 8d 0d d2 18 00 00 	lea    0x18d2(%rip),%rcx        # 403a40 <_ZN2Vc2v03AVX18_IndexesFromZero32E>
  40216e:	4c 8d 05 97 15 00 00 	lea    0x1597(%rip),%r8        # 40370c <_ZN2Vc2v03AVX9c_general8oneFloatE>
  402175:	c4 e2 7d 18 50 04    	vbroadcastss 0x4(%rax),%ymm2
  40217b:	48 8d 05 be 15 00 00 	lea    0x15be(%rip),%rax        # 403740 <_ZN2Vc2v03AVX6c_trigIfE4dataE>
  402182:	c4 e2 7d 18 61 04    	vbroadcastss 0x4(%rcx),%ymm4
  402188:	c5 ec 54 17          	vandps (%rdi),%ymm2,%ymm2
  40218c:	c4 e2 7d 18 40 54    	vbroadcastss 0x54(%rax),%ymm0
  402192:	c4 62 7d 18 48 24    	vbroadcastss 0x24(%rax),%ymm9
  402198:	c4 62 7d 18 40 20    	vbroadcastss 0x20(%rax),%ymm8
  40219e:	c5 ec 59 c0          	vmulps %ymm0,%ymm2,%ymm0
  4021a2:	c5 fe 5b c0          	vcvttps2dq %ymm0,%ymm0
  4021a6:	c5 dc 54 c8          	vandps %ymm0,%ymm4,%ymm1
  4021aa:	c4 e3 7d 19 c6 01    	vextractf128 $0x1,%ymm0,%xmm6
  4021b0:	c5 f9 6f e8          	vmovdqa %xmm0,%xmm5
  4021b4:	c4 e3 7d 19 cb 01    	vextractf128 $0x1,%ymm1,%xmm3
  4021ba:	c5 d1 fe e9          	vpaddd %xmm1,%xmm5,%xmm5
  4021be:	c4 e2 7d 18 48 04    	vbroadcastss 0x4(%rax),%ymm1
  4021c4:	c5 c9 fe db          	vpaddd %xmm3,%xmm6,%xmm3
  4021c8:	c4 e3 55 18 eb 01    	vinsertf128 $0x1,%xmm3,%ymm5,%ymm5
  4021ce:	c4 e2 7d 18 58 08    	vbroadcastss 0x8(%rax),%ymm3
  4021d4:	c5 fc 5b c5          	vcvtdq2ps %ymm5,%ymm0
  4021d8:	c5 fc 59 c9          	vmulps %ymm1,%ymm0,%ymm1
  4021dc:	c5 fc 59 db          	vmulps %ymm3,%ymm0,%ymm3
  4021e0:	c5 d4 54 2d 78 18 00 	vandps 0x1878(%rip),%ymm5,%ymm5        # 403a60 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x20>
  4021e7:	00 
  4021e8:	c5 ec 5c d1          	vsubps %ymm1,%ymm2,%ymm2
  4021ec:	c5 ec 5c cb          	vsubps %ymm3,%ymm2,%ymm1
  4021f0:	c4 e2 7d 18 50 0c    	vbroadcastss 0xc(%rax),%ymm2
  4021f6:	c5 fd 6f 1d 82 18 00 	vmovdqa 0x1882(%rip),%ymm3        # 403a80 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x40>
  4021fd:	00 
  4021fe:	c5 fc 59 c2          	vmulps %ymm2,%ymm0,%ymm0
  402202:	c4 e3 7d 19 db 01    	vextractf128 $0x1,%ymm3,%xmm3
  402208:	c5 f4 5c c8          	vsubps %ymm0,%ymm1,%ymm1
  40220c:	c4 e3 7d 19 e8 01    	vextractf128 $0x1,%ymm5,%xmm0
  402212:	c5 d1 66 d3          	vpcmpgtd %xmm3,%xmm5,%xmm2
  402216:	c5 f9 66 db          	vpcmpgtd %xmm3,%xmm0,%xmm3
  40221a:	c4 e3 6d 18 db 01    	vinsertf128 $0x1,%xmm3,%ymm2,%ymm3
  402220:	c5 e4 54 35 78 18 00 	vandps 0x1878(%rip),%ymm3,%ymm6        # 403aa0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x60>
  402227:	00 
  402228:	c4 e3 7d 19 f7 01    	vextractf128 $0x1,%ymm6,%xmm7
  40222e:	c5 d1 fa ee          	vpsubd %xmm6,%xmm5,%xmm5
  402232:	c4 c2 7d 18 30       	vbroadcastss (%r8),%ymm6
  402237:	c5 f9 6f d5          	vmovdqa %xmm5,%xmm2
  40223b:	c4 e2 7d 18 68 28    	vbroadcastss 0x28(%rax),%ymm5
  402241:	4c 8d 05 78 13 00 00 	lea    0x1378(%rip),%r8        # 4035c0 <_ZN2Vc2v03AVX5c_logIfE4dataE>
  402248:	c5 f9 fa c7          	vpsubd %xmm7,%xmm0,%xmm0
  40224c:	c4 e3 6d 18 d0 01    	vinsertf128 $0x1,%xmm0,%ymm2,%ymm2
  402252:	c5 f4 59 c1          	vmulps %ymm1,%ymm1,%ymm0
  402256:	c5 fc 59 ed          	vmulps %ymm5,%ymm0,%ymm5
  40225a:	c5 fc 59 f8          	vmulps %ymm0,%ymm0,%ymm7
  40225e:	c5 b4 58 ed          	vaddps %ymm5,%ymm9,%ymm5
  402262:	c4 62 7d 18 48 3c    	vbroadcastss 0x3c(%rax),%ymm9
  402268:	c5 fc 59 ed          	vmulps %ymm5,%ymm0,%ymm5
  40226c:	c5 bc 58 ed          	vaddps %ymm5,%ymm8,%ymm5
  402270:	c4 62 7d 18 40 40    	vbroadcastss 0x40(%rax),%ymm8
  402276:	c4 41 7c 59 c0       	vmulps %ymm8,%ymm0,%ymm8
  40227b:	c5 c4 59 ed          	vmulps %ymm5,%ymm7,%ymm5
  40227f:	c4 c2 7d 18 78 48    	vbroadcastss 0x48(%r8),%ymm7
  402285:	c5 fc 59 ff          	vmulps %ymm7,%ymm0,%ymm7
  402289:	c4 41 34 58 c0       	vaddps %ymm8,%ymm9,%ymm8
  40228e:	c5 d4 5c ef          	vsubps %ymm7,%ymm5,%ymm5
  402292:	c4 e2 7d 18 78 38    	vbroadcastss 0x38(%rax),%ymm7
  402298:	c5 cc 58 ed          	vaddps %ymm5,%ymm6,%ymm5
  40229c:	c5 f4 59 f0          	vmulps %ymm0,%ymm1,%ymm6
  4022a0:	c4 c1 7c 59 c0       	vmulps %ymm8,%ymm0,%ymm0
  4022a5:	c4 c3 7d 19 e0 01    	vextractf128 $0x1,%ymm4,%xmm8
  4022ab:	c5 c4 58 c0          	vaddps %ymm0,%ymm7,%ymm0
  4022af:	c5 cc 59 c0          	vmulps %ymm0,%ymm6,%ymm0
  4022b3:	c5 fd 6f 35 05 18 00 	vmovdqa 0x1805(%rip),%ymm6        # 403ac0 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0x80>
  4022ba:	00 
  4022bb:	c4 e3 7d 19 f6 01    	vextractf128 $0x1,%ymm6,%xmm6
  4022c1:	c5 f4 58 c0          	vaddps %ymm0,%ymm1,%ymm0
  4022c5:	c4 e3 7d 19 d1 01    	vextractf128 $0x1,%ymm2,%xmm1
  4022cb:	c5 e9 76 fe          	vpcmpeqd %xmm6,%xmm2,%xmm7
  4022cf:	c5 f1 76 f6          	vpcmpeqd %xmm6,%xmm1,%xmm6
  4022d3:	c4 41 71 76 c8       	vpcmpeqd %xmm8,%xmm1,%xmm9
  4022d8:	c4 e3 45 18 f6 01    	vinsertf128 $0x1,%xmm6,%ymm7,%ymm6
  4022de:	c5 e9 76 fc          	vpcmpeqd %xmm4,%xmm2,%xmm7
  4022e2:	c4 41 71 66 c0       	vpcmpgtd %xmm8,%xmm1,%xmm8
  4022e7:	c4 c3 45 18 f9 01    	vinsertf128 $0x1,%xmm9,%ymm7,%ymm7
  4022ed:	c5 e9 66 e4          	vpcmpgtd %xmm4,%xmm2,%xmm4
  4022f1:	c4 43 5d 18 c0 01    	vinsertf128 $0x1,%xmm8,%ymm4,%ymm8
  4022f7:	c5 fc 28 25 01 18 00 	vmovaps 0x1801(%rip),%ymm4        # 403b00 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xc0>
  4022fe:	00 
  4022ff:	c5 c4 56 fe          	vorps  %ymm6,%ymm7,%ymm7
  402303:	c4 41 64 57 c0       	vxorps %ymm8,%ymm3,%ymm8
  402308:	c4 e3 55 4a f8 70    	vblendvps %ymm7,%ymm0,%ymm5,%ymm7
  40230e:	c5 44 57 cc          	vxorps %ymm4,%ymm7,%ymm9
  402312:	c4 c3 45 4a f9 80    	vblendvps %ymm8,%ymm9,%ymm7,%ymm7
  402318:	c5 fc 29 3a          	vmovaps %ymm7,(%rdx)
  40231c:	c4 e2 7d 18 79 04    	vbroadcastss 0x4(%rcx),%ymm7
  402322:	c4 c3 7d 19 f8 01    	vextractf128 $0x1,%ymm7,%xmm8
  402328:	c5 e9 76 d7          	vpcmpeqd %xmm7,%xmm2,%xmm2
  40232c:	c4 c1 71 76 c8       	vpcmpeqd %xmm8,%xmm1,%xmm1
  402331:	c4 e3 6d 18 c9 01    	vinsertf128 $0x1,%xmm1,%ymm2,%ymm1
  402337:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
  40233b:	c5 f4 56 ce          	vorps  %ymm6,%ymm1,%ymm1
  40233f:	c4 e3 7d 4a cd 10    	vblendvps %ymm1,%ymm5,%ymm0,%ymm1
  402345:	c5 fc 28 07          	vmovaps (%rdi),%ymm0
  402349:	c5 fc c2 c2 01       	vcmpltps %ymm2,%ymm0,%ymm0
  40234e:	c5 f4 57 e4          	vxorps %ymm4,%ymm1,%ymm4
  402352:	c5 fc 57 db          	vxorps %ymm3,%ymm0,%ymm3
  402356:	c4 e3 75 4a cc 30    	vblendvps %ymm3,%ymm4,%ymm1,%ymm1
  40235c:	c5 fc 29 0e          	vmovaps %ymm1,(%rsi)
  402360:	c5 f8 77             	vzeroupper 
  402363:	c3                   	retq   
  402364:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40236b:	00 00 00 
  40236e:	66 90                	xchg   %ax,%ax

0000000000402370 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE4asinIfEENS0_3AVX6VectorIT_EERKSB_>:
  402370:	48 8d 05 a5 13 00 00 	lea    0x13a5(%rip),%rax        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  402377:	c5 fc 28 27          	vmovaps (%rdi),%ymm4
  40237b:	48 8d 15 3e 12 00 00 	lea    0x123e(%rip),%rdx        # 4035c0 <_ZN2Vc2v03AVX5c_logIfE4dataE>
  402382:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  402386:	c4 e2 7d 18 48 04    	vbroadcastss 0x4(%rax),%ymm1
  40238c:	48 8d 05 79 13 00 00 	lea    0x1379(%rip),%rax        # 40370c <_ZN2Vc2v03AVX9c_general8oneFloatE>
  402393:	c5 dc c2 c0 01       	vcmpltps %ymm0,%ymm4,%ymm0
  402398:	c5 dc 54 c9          	vandps %ymm1,%ymm4,%ymm1
  40239c:	c4 e2 7d 18 62 48    	vbroadcastss 0x48(%rdx),%ymm4
  4023a2:	c4 e2 7d 18 28       	vbroadcastss (%rax),%ymm5
  4023a7:	48 8d 05 92 13 00 00 	lea    0x1392(%rip),%rax        # 403740 <_ZN2Vc2v03AVX6c_trigIfE4dataE>
  4023ae:	c5 d4 5c d1          	vsubps %ymm1,%ymm5,%ymm2
  4023b2:	c5 74 59 c1          	vmulps %ymm1,%ymm1,%ymm8
  4023b6:	c4 62 7d 18 98 a4 00 	vbroadcastss 0xa4(%rax),%ymm11
  4023bd:	00 00 
  4023bf:	c5 f4 c2 dc 06       	vcmpnleps %ymm4,%ymm1,%ymm3
  4023c4:	c4 62 7d 18 90 a8 00 	vbroadcastss 0xa8(%rax),%ymm10
  4023cb:	00 00 
  4023cd:	c4 62 7d 18 88 ac 00 	vbroadcastss 0xac(%rax),%ymm9
  4023d4:	00 00 
  4023d6:	c5 f4 c2 f5 06       	vcmpnleps %ymm5,%ymm1,%ymm6
  4023db:	c4 e2 7d 18 a8 a0 00 	vbroadcastss 0xa0(%rax),%ymm5
  4023e2:	00 00 
  4023e4:	c5 dc 59 d2          	vmulps %ymm2,%ymm4,%ymm2
  4023e8:	c4 e2 7d 18 b8 94 00 	vbroadcastss 0x94(%rax),%ymm7
  4023ef:	00 00 
  4023f1:	c5 f4 c2 ff 01       	vcmpltps %ymm7,%ymm1,%ymm7
  4023f6:	c4 e3 3d 4a d2 30    	vblendvps %ymm3,%ymm2,%ymm8,%ymm2
  4023fc:	c4 62 7d 18 80 b0 00 	vbroadcastss 0xb0(%rax),%ymm8
  402403:	00 00 
  402405:	c5 ec 59 ed          	vmulps %ymm5,%ymm2,%ymm5
  402409:	c5 fc 51 e2          	vsqrtps %ymm2,%ymm4
  40240d:	c4 e3 75 4a e4 30    	vblendvps %ymm3,%ymm4,%ymm1,%ymm4
  402413:	c5 a4 58 ed          	vaddps %ymm5,%ymm11,%ymm5
  402417:	c5 ec 59 ed          	vmulps %ymm5,%ymm2,%ymm5
  40241b:	c5 ac 58 ed          	vaddps %ymm5,%ymm10,%ymm5
  40241f:	c5 ec 59 ed          	vmulps %ymm5,%ymm2,%ymm5
  402423:	c5 b4 58 ed          	vaddps %ymm5,%ymm9,%ymm5
  402427:	c5 ec 59 ed          	vmulps %ymm5,%ymm2,%ymm5
  40242b:	c5 bc 58 ed          	vaddps %ymm5,%ymm8,%ymm5
  40242f:	c5 ec 59 d5          	vmulps %ymm5,%ymm2,%ymm2
  402433:	c4 e2 7d 18 68 58    	vbroadcastss 0x58(%rax),%ymm5
  402439:	c5 dc 59 d2          	vmulps %ymm2,%ymm4,%ymm2
  40243d:	c5 dc 58 d2          	vaddps %ymm2,%ymm4,%ymm2
  402441:	c5 ec 58 e2          	vaddps %ymm2,%ymm2,%ymm4
  402445:	c5 d4 5c e4          	vsubps %ymm4,%ymm5,%ymm4
  402449:	c4 e3 6d 4a d4 30    	vblendvps %ymm3,%ymm4,%ymm2,%ymm2
  40244f:	c4 e3 6d 4a c9 70    	vblendvps %ymm7,%ymm1,%ymm2,%ymm1
  402455:	c5 f4 57 15 a3 16 00 	vxorps 0x16a3(%rip),%ymm1,%ymm2        # 403b00 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xc0>
  40245c:	00 
  40245d:	c4 e3 75 4a ca 00    	vblendvps %ymm0,%ymm2,%ymm1,%ymm1
  402463:	c5 f4 56 c6          	vorps  %ymm6,%ymm1,%ymm0
  402467:	c3                   	retq   
  402468:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40246f:	00 

0000000000402470 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE4atanIfEENS0_3AVX6VectorIT_EERKSB_>:
  402470:	48 8d 05 a5 12 00 00 	lea    0x12a5(%rip),%rax        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  402477:	c5 fc 28 0f          	vmovaps (%rdi),%ymm1
  40247b:	48 8d 15 1e 11 00 00 	lea    0x111e(%rip),%rdx        # 4035a0 <_ZN2Vc2v06Common10AllBitsSetE>
  402482:	c5 c8 57 f6          	vxorps %xmm6,%xmm6,%xmm6
  402486:	c5 fc 28 2d 72 16 00 	vmovaps 0x1672(%rip),%ymm5        # 403b00 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xc0>
  40248d:	00 
  40248e:	c4 e2 7d 18 50 04    	vbroadcastss 0x4(%rax),%ymm2
  402494:	48 8d 05 a5 12 00 00 	lea    0x12a5(%rip),%rax        # 403740 <_ZN2Vc2v03AVX6c_trigIfE4dataE>
  40249b:	c5 f4 54 d2          	vandps %ymm2,%ymm1,%ymm2
  40249f:	c4 e2 7d 18 a0 88 00 	vbroadcastss 0x88(%rax),%ymm4
  4024a6:	00 00 
  4024a8:	c4 e2 7d 18 b8 8c 00 	vbroadcastss 0x8c(%rax),%ymm7
  4024af:	00 00 
  4024b1:	c5 ec c2 e4 06       	vcmpnleps %ymm4,%ymm2,%ymm4
  4024b6:	c4 62 7d 18 40 58    	vbroadcastss 0x58(%rax),%ymm8
  4024bc:	c4 e2 7d 18 00       	vbroadcastss (%rax),%ymm0
  4024c1:	c5 ec c2 ff 06       	vcmpnleps %ymm7,%ymm2,%ymm7
  4024c6:	c5 dc 55 1a          	vandnps (%rdx),%ymm4,%ymm3
  4024ca:	48 8d 15 3b 12 00 00 	lea    0x123b(%rip),%rdx        # 40370c <_ZN2Vc2v03AVX9c_general8oneFloatE>
  4024d1:	c4 43 4d 4a c0 40    	vblendvps %ymm4,%ymm8,%ymm6,%ymm8
  4024d7:	c5 f4 c2 f6 01       	vcmpltps %ymm6,%ymm1,%ymm6
  4024dc:	c5 f4 c2 c9 03       	vcmpunordps %ymm1,%ymm1,%ymm1
  4024e1:	c5 c4 54 fb          	vandps %ymm3,%ymm7,%ymm7
  4024e5:	c4 e2 7d 18 1a       	vbroadcastss (%rdx),%ymm3
  4024ea:	c4 63 3d 4a c0 70    	vblendvps %ymm7,%ymm0,%ymm8,%ymm8
  4024f0:	c5 e4 57 c5          	vxorps %ymm5,%ymm3,%ymm0
  4024f4:	c5 fc 5e c2          	vdivps %ymm2,%ymm0,%ymm0
  4024f8:	c4 e3 6d 4a d0 40    	vblendvps %ymm4,%ymm0,%ymm2,%ymm2
  4024fe:	c4 e2 7d 18 60 68    	vbroadcastss 0x68(%rax),%ymm4
  402504:	c5 ec 5c c3          	vsubps %ymm3,%ymm2,%ymm0
  402508:	c5 ec 58 db          	vaddps %ymm3,%ymm2,%ymm3
  40250c:	c5 fc 5e c3          	vdivps %ymm3,%ymm0,%ymm0
  402510:	c4 e2 7d 18 58 60    	vbroadcastss 0x60(%rax),%ymm3
  402516:	c4 e3 6d 4a c0 70    	vblendvps %ymm7,%ymm0,%ymm2,%ymm0
  40251c:	c4 e2 7d 18 78 64    	vbroadcastss 0x64(%rax),%ymm7
  402522:	c5 fc 59 d0          	vmulps %ymm0,%ymm0,%ymm2
  402526:	c5 ec 59 db          	vmulps %ymm3,%ymm2,%ymm3
  40252a:	c5 e4 5c df          	vsubps %ymm7,%ymm3,%ymm3
  40252e:	c5 ec 59 db          	vmulps %ymm3,%ymm2,%ymm3
  402532:	c5 dc 58 db          	vaddps %ymm3,%ymm4,%ymm3
  402536:	c4 e2 7d 18 60 6c    	vbroadcastss 0x6c(%rax),%ymm4
  40253c:	c5 ec 59 db          	vmulps %ymm3,%ymm2,%ymm3
  402540:	c5 e4 5c dc          	vsubps %ymm4,%ymm3,%ymm3
  402544:	c5 ec 59 d3          	vmulps %ymm3,%ymm2,%ymm2
  402548:	c5 fc 59 d2          	vmulps %ymm2,%ymm0,%ymm2
  40254c:	c5 fc 58 c2          	vaddps %ymm2,%ymm0,%ymm0
  402550:	c5 bc 58 c0          	vaddps %ymm0,%ymm8,%ymm0
  402554:	c5 fc 57 ed          	vxorps %ymm5,%ymm0,%ymm5
  402558:	c4 e3 7d 4a c5 60    	vblendvps %ymm6,%ymm5,%ymm0,%ymm0
  40255e:	c5 fc 56 c1          	vorps  %ymm1,%ymm0,%ymm0
  402562:	c3                   	retq   
  402563:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40256a:	00 00 00 
  40256d:	0f 1f 00             	nopl   (%rax)

0000000000402570 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE5atan2IfEENS0_3AVX6VectorIT_EERKSB_SD_>:
  402570:	55                   	push   %rbp
  402571:	c5 f0 57 c9          	vxorps %xmm1,%xmm1,%xmm1
  402575:	48 89 e5             	mov    %rsp,%rbp
  402578:	41 57                	push   %r15
  40257a:	49 89 ff             	mov    %rdi,%r15
  40257d:	41 56                	push   %r14
  40257f:	41 55                	push   %r13
  402581:	41 54                	push   %r12
  402583:	49 89 f4             	mov    %rsi,%r12
  402586:	53                   	push   %rbx
  402587:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40258b:	48 81 ec 40 01 00 00 	sub    $0x140,%rsp
  402592:	4c 8d 2d 7b 11 00 00 	lea    0x117b(%rip),%r13        # 403714 <_ZN2Vc2v03AVX9c_general13signMaskFloatE>
  402599:	48 8d 05 00 10 00 00 	lea    0x1000(%rip),%rax        # 4035a0 <_ZN2Vc2v06Common10AllBitsSetE>
  4025a0:	c5 fc 28 07          	vmovaps (%rdi),%ymm0
  4025a4:	48 8d 1d 95 11 00 00 	lea    0x1195(%rip),%rbx        # 403740 <_ZN2Vc2v03AVX6c_trigIfE4dataE>
  4025ab:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  4025b2:	00 
  4025b3:	c5 fc 28 16          	vmovaps (%rsi),%ymm2
  4025b7:	4c 8d 35 5e 11 00 00 	lea    0x115e(%rip),%r14        # 40371c <_ZN2Vc2v03AVX9c_general12absMaskFloatE>
  4025be:	c5 fc c2 e9 00       	vcmpeqps %ymm1,%ymm0,%ymm5
  4025c3:	c4 62 7d 18 4b 5c    	vbroadcastss 0x5c(%rbx),%ymm9
  4025c9:	c5 ec c2 e1 00       	vcmpeqps %ymm1,%ymm2,%ymm4
  4025ce:	c5 fc 29 ac 24 00 01 	vmovaps %ymm5,0x100(%rsp)
  4025d5:	00 00 
  4025d7:	c4 42 7d 18 45 04    	vbroadcastss 0x4(%r13),%ymm8
  4025dd:	c5 fc 29 24 24       	vmovaps %ymm4,(%rsp)
  4025e2:	c5 bc 54 ea          	vandps %ymm2,%ymm8,%ymm5
  4025e6:	c4 41 7c 54 c0       	vandps %ymm8,%ymm0,%ymm8
  4025eb:	c4 e3 7d 19 ee 01    	vextractf128 $0x1,%ymm5,%xmm6
  4025f1:	c5 f9 6f dd          	vmovdqa %xmm5,%xmm3
  4025f5:	c5 c9 72 e6 1f       	vpsrad $0x1f,%xmm6,%xmm6
  4025fa:	c5 e1 72 e3 1f       	vpsrad $0x1f,%xmm3,%xmm3
  4025ff:	c4 e3 65 18 de 01    	vinsertf128 $0x1,%xmm6,%ymm3,%ymm3
  402605:	c5 ec 59 f1          	vmulps %ymm1,%ymm2,%ymm6
  402609:	c5 dc 54 fb          	vandps %ymm3,%ymm4,%ymm7
  40260d:	c5 fc 28 18          	vmovaps (%rax),%ymm3
  402611:	c5 fc c2 e1 01       	vcmpltps %ymm1,%ymm0,%ymm4
  402616:	48 8d 05 ef 10 00 00 	lea    0x10ef(%rip),%rax        # 40370c <_ZN2Vc2v03AVX9c_general8oneFloatE>
  40261d:	c5 fc 29 5c 24 20    	vmovaps %ymm3,0x20(%rsp)
  402623:	c5 ec c2 f6 07       	vcmpordps %ymm6,%ymm2,%ymm6
  402628:	c5 fc 29 bc 24 e0 00 	vmovaps %ymm7,0xe0(%rsp)
  40262f:	00 00 
  402631:	c5 fc 29 a4 24 c0 00 	vmovaps %ymm4,0xc0(%rsp)
  402638:	00 00 
  40263a:	c5 cc 55 f3          	vandnps %ymm3,%ymm6,%ymm6
  40263e:	c5 fc 29 b4 24 a0 00 	vmovaps %ymm6,0xa0(%rsp)
  402645:	00 00 
  402647:	c5 fc 59 f1          	vmulps %ymm1,%ymm0,%ymm6
  40264b:	c4 c2 7d 18 7e 04    	vbroadcastss 0x4(%r14),%ymm7
  402651:	c5 fc 29 4c 24 40    	vmovaps %ymm1,0x40(%rsp)
  402657:	c5 34 54 cf          	vandps %ymm7,%ymm9,%ymm9
  40265b:	c5 fc c2 f6 07       	vcmpordps %ymm6,%ymm0,%ymm6
  402660:	c4 41 3c 56 c1       	vorps  %ymm9,%ymm8,%ymm8
  402665:	c5 6c c2 c9 05       	vcmpnltps %ymm1,%ymm2,%ymm9
  40266a:	c5 cc 55 f3          	vandnps %ymm3,%ymm6,%ymm6
  40266e:	c4 c1 34 55 c8       	vandnps %ymm8,%ymm9,%ymm1
  402673:	c5 fc 29 8c 24 80 00 	vmovaps %ymm1,0x80(%rsp)
  40267a:	00 00 
  40267c:	c4 62 7d 18 00       	vbroadcastss (%rax),%ymm8
  402681:	c5 fc 29 74 24 60    	vmovaps %ymm6,0x60(%rsp)
  402687:	c5 bc 54 ff          	vandps %ymm7,%ymm8,%ymm7
  40268b:	c5 d4 56 ef          	vorps  %ymm7,%ymm5,%ymm5
  40268f:	c4 e3 6d 4a d5 60    	vblendvps %ymm6,%ymm5,%ymm2,%ymm2
  402695:	c5 fc 5e c2          	vdivps %ymm2,%ymm0,%ymm0
  402699:	c5 fc 29 84 24 20 01 	vmovaps %ymm0,0x120(%rsp)
  4026a0:	00 00 
  4026a2:	c5 f8 77             	vzeroupper 
  4026a5:	e8 c6 fd ff ff       	callq  402470 <_ZN2Vc2v06Common13TrigonometricINS0_6Public15ImplementationTILj6EEEE4atanIfEENS0_3AVX6VectorIT_EERKSB_>
  4026aa:	c5 fc 28 24 24       	vmovaps (%rsp),%ymm4
  4026af:	c5 fc 58 84 24 80 00 	vaddps 0x80(%rsp),%ymm0,%ymm0
  4026b6:	00 00 
  4026b8:	c5 5c 54 84 24 00 01 	vandps 0x100(%rsp),%ymm4,%ymm8
  4026bf:	00 00 
  4026c1:	c4 c1 7c 28 17       	vmovaps (%r15),%ymm2
  4026c6:	c4 c2 7d 18 6e 04    	vbroadcastss 0x4(%r14),%ymm5
  4026cc:	c4 c2 7d 18 7d 04    	vbroadcastss 0x4(%r13),%ymm7
  4026d2:	c4 62 7d 18 4b 5c    	vbroadcastss 0x5c(%rbx),%ymm9
  4026d8:	c5 fc 28 5c 24 20    	vmovaps 0x20(%rsp),%ymm3
  4026de:	c5 3c 55 c0          	vandnps %ymm0,%ymm8,%ymm8
  4026e2:	c5 fc 28 74 24 60    	vmovaps 0x60(%rsp),%ymm6
  4026e8:	c5 ec 54 c7          	vandps %ymm7,%ymm2,%ymm0
  4026ec:	c5 fc 28 4c 24 40    	vmovaps 0x40(%rsp),%ymm1
  4026f2:	c5 34 54 cd          	vandps %ymm5,%ymm9,%ymm9
  4026f6:	c5 ec 55 db          	vandnps %ymm3,%ymm2,%ymm3
  4026fa:	c5 ec c2 d2 03       	vcmpunordps %ymm2,%ymm2,%ymm2
  4026ff:	c5 dc 54 a4 24 c0 00 	vandps 0xc0(%rsp),%ymm4,%ymm4
  402706:	00 00 
  402708:	c4 41 7c 56 c9       	vorps  %ymm9,%ymm0,%ymm9
  40270d:	c4 c1 64 57 1c 24    	vxorps (%r12),%ymm3,%ymm3
  402713:	c5 34 54 8c 24 e0 00 	vandps 0xe0(%rsp),%ymm9,%ymm9
  40271a:	00 00 
  40271c:	c5 cc 54 b4 24 a0 00 	vandps 0xa0(%rsp),%ymm6,%ymm6
  402723:	00 00 
  402725:	c5 e4 54 ff          	vandps %ymm7,%ymm3,%ymm7
  402729:	c4 c1 7c 28 1c 24    	vmovaps (%r12),%ymm3
  40272f:	c4 41 3c 58 c1       	vaddps %ymm9,%ymm8,%ymm8
  402734:	c4 62 7d 18 4b 58    	vbroadcastss 0x58(%rbx),%ymm9
  40273a:	c5 34 57 0d be 13 00 	vxorps 0x13be(%rip),%ymm9,%ymm9        # 403b00 <_ZN2Vc2v03AVX18_IndexesFromZero32E+0xc0>
  402741:	00 
  402742:	c5 e4 c2 db 03       	vcmpunordps %ymm3,%ymm3,%ymm3
  402747:	c5 ec 56 d3          	vorps  %ymm3,%ymm2,%ymm2
  40274b:	c4 c3 3d 4a e1 40    	vblendvps %ymm4,%ymm9,%ymm8,%ymm4
  402751:	c4 62 7d 18 03       	vbroadcastss (%rbx),%ymm8
  402756:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  40275a:	c5 3c 54 c5          	vandps %ymm5,%ymm8,%ymm8
  40275e:	5b                   	pop    %rbx
  40275f:	41 5c                	pop    %r12
  402761:	c4 c1 44 56 f8       	vorps  %ymm8,%ymm7,%ymm7
  402766:	41 5d                	pop    %r13
  402768:	41 5e                	pop    %r14
  40276a:	c5 cc 54 f7          	vandps %ymm7,%ymm6,%ymm6
  40276e:	41 5f                	pop    %r15
  402770:	5d                   	pop    %rbp
  402771:	c5 dc 58 e6          	vaddps %ymm6,%ymm4,%ymm4
  402775:	c5 dc 54 ed          	vandps %ymm5,%ymm4,%ymm5
  402779:	c5 dc c2 c9 00       	vcmpeqps %ymm1,%ymm4,%ymm1
  40277e:	c5 fc 56 c5          	vorps  %ymm5,%ymm0,%ymm0
  402782:	c4 e3 5d 4a e0 10    	vblendvps %ymm1,%ymm0,%ymm4,%ymm4
  402788:	c5 dc 56 c2          	vorps  %ymm2,%ymm4,%ymm0
  40278c:	c3                   	retq   
  40278d:	0f 1f 00             	nopl   (%rax)

0000000000402790 <__libc_csu_fini>:
  402790:	f3 c3                	repz retq 
  402792:	66 66 66 66 66 2e 0f 	data32 data32 data32 data32 nopw %cs:0x0(%rax,%rax,1)
  402799:	1f 84 00 00 00 00 00 

00000000004027a0 <__libc_csu_init>:
  4027a0:	48 89 6c 24 d8       	mov    %rbp,-0x28(%rsp)
  4027a5:	4c 89 64 24 e0       	mov    %r12,-0x20(%rsp)
  4027aa:	48 8d 2d 67 18 20 00 	lea    0x201867(%rip),%rbp        # 604018 <__init_array_end>
  4027b1:	4c 8d 25 48 18 20 00 	lea    0x201848(%rip),%r12        # 604000 <__frame_dummy_init_array_entry>
  4027b8:	4c 89 6c 24 e8       	mov    %r13,-0x18(%rsp)
  4027bd:	4c 89 74 24 f0       	mov    %r14,-0x10(%rsp)
  4027c2:	4c 89 7c 24 f8       	mov    %r15,-0x8(%rsp)
  4027c7:	48 89 5c 24 d0       	mov    %rbx,-0x30(%rsp)
  4027cc:	48 83 ec 38          	sub    $0x38,%rsp
  4027d0:	4c 29 e5             	sub    %r12,%rbp
  4027d3:	41 89 fd             	mov    %edi,%r13d
  4027d6:	49 89 f6             	mov    %rsi,%r14
  4027d9:	48 c1 fd 03          	sar    $0x3,%rbp
  4027dd:	49 89 d7             	mov    %rdx,%r15
  4027e0:	e8 73 e0 ff ff       	callq  400858 <_init>
  4027e5:	48 85 ed             	test   %rbp,%rbp
  4027e8:	74 1c                	je     402806 <__libc_csu_init+0x66>
  4027ea:	31 db                	xor    %ebx,%ebx
  4027ec:	0f 1f 40 00          	nopl   0x0(%rax)
  4027f0:	4c 89 fa             	mov    %r15,%rdx
  4027f3:	4c 89 f6             	mov    %r14,%rsi
  4027f6:	44 89 ef             	mov    %r13d,%edi
  4027f9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4027fd:	48 83 c3 01          	add    $0x1,%rbx
  402801:	48 39 eb             	cmp    %rbp,%rbx
  402804:	72 ea                	jb     4027f0 <__libc_csu_init+0x50>
  402806:	48 8b 5c 24 08       	mov    0x8(%rsp),%rbx
  40280b:	48 8b 6c 24 10       	mov    0x10(%rsp),%rbp
  402810:	4c 8b 64 24 18       	mov    0x18(%rsp),%r12
  402815:	4c 8b 6c 24 20       	mov    0x20(%rsp),%r13
  40281a:	4c 8b 74 24 28       	mov    0x28(%rsp),%r14
  40281f:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
  402824:	48 83 c4 38          	add    $0x38,%rsp
  402828:	c3                   	retq   
  402829:	0f 1f 00             	nopl   (%rax)

Disassembly of section .fini:

000000000040282c <_fini>:
  40282c:	48 83 ec 08          	sub    $0x8,%rsp
  402830:	48 83 c4 08          	add    $0x8,%rsp
  402834:	c3                   	retq   
