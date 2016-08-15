

%% build tree
disp('build tree')
b   = fvecs_read('/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/base.fvecs');
b   = b(:,1:100000);
hnd = pqt('build',b);
pqt('save_tree', hnd, 'test.tree')
pqt('destroy',hnd);

% each line of b is one vector


%% insert db
disp('build db')
hnd   = pqt('read_tree','test.tree');
b     = fvecs_read('/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/base.fvecs');
b     = b';
[n,d] = size(b);
pqt('notify', hnd, n) % we need to tell the pqt in advance how many vectors there are at max
for i=1:n
	vec = b(i,:);
	pqt('insert', hnd, vec);
end
pqt('save_bins', hnd, 'test.bins')
pqt('destroy',hnd);

%% query
% do:    hnd    = pqt('init','test.tree','test.bins');
% or
hnd = pqt('read_tree','test.tree');
pqt('read_bins',hnd, 'test.bins');

v      = fvecs_read('/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/query.fvecs');
g      = ivecs_read('/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/groundtruth.ivecs');


recall = zeros(5,1);
tic
for i=1:1000
	q = v(:,i)';
	[ids,dst] = pqt('query',hnd,q);
	
    clen = numel(ids);
    
	if numel(find(ids(1:min(1,clen))==g(1,i)))
		recall(1) = recall(1) + 1;
	end
	if numel(find(ids(1:min(10,clen))==g(1,i)))
		recall(2) = recall(2) + 1;
	end
	if numel(find(ids(1:min(100,clen))==g(1,i)))
		recall(3) = recall(3) + 1;
	end
	if numel(find(ids(1:min(1000,clen))==g(1,i)))
		recall(4) = recall(4) + 1;
	end
	if numel(find(ids(1:min(10000,clen))==g(1,i)))
		recall(5) = recall(5) + 1;
	end
end
toc/1000
recall = recall / 1000;


pqt('destroy',hnd);

