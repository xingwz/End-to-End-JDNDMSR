function bayer = mosaicrut(b)
  [y,x,k] = size(b);
  bayer = zeros(y,x); 
  m2 = [1 2; 2 3];
  for i=1:y
    for j=1:x
      i1=2-mod(i,2);
      j1=2-mod(j,2);
      bayer(i,j)=b(i,j,m2(i1,j1));
    end
  end
end
