import collections


for i in range(5):
    print(i)

def minDistance(word1,word2):
    n1=len(word1)
    n2=len(word2)
    dp=[]
    for i in range(n1+1):
        dp.append([0]*(n2+1))
    for j in range(1,n2+1):
        dp[0][j]=dp[0][j-1]+1
    for i in range(1,n1+1):
        dp[i][0]=dp[i-1][0]+1
    for i in range(1,n1+1):
        for j in range(1,n2+1):
            if word1[i-1]==word2[j-1]:
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=min(dp[i-1][j-1],dp[i][j-1],dp[i-1][j])+1

    return dp[n1][n2]

def minDistance(word1,word2):
    m=len(word1)
    n=len(word2)
    dp=[]
    for i in range(m+1):
        dp.append([0]*(n+1))

def ismatch(s,p):
    m=len(s)
    n=len(p)
    dp=[]
    for i in range(m+1):
        dp.append([False]*(n+1))
    dp[0][0]=True
    for i in range(1,m+1):
        dp[i][0]=False
    star=1
    for j in range(1,n+1):
        if p[j-1]=='*' and star==1:
            dp[0][j]=True
        else:
            star=0
            dp[0][j]=False
    for i in range(1,m+1):
        for j in range(1,n+1):
            if p[j-1]=='?' or s[i-1]==p[j-1]:
                dp[i][j]=dp[i-1][j-1]
            elif p[j-1]=='*':
                dp[i][j]=dp[i-1][j]|dp[i][j-1]
    return dp[m][n]



# print(ismatch('text','text'))
# print(minDistance('text','text'))

def minWindow(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    hash_t = collections.Counter(t)
    m = len(s)
    n = len(t)
    lenth = max(m, n) + 1
    need_str = n
    left_point = 0
    min_str = ''
    for i in range(m):
        in_str = s[i]
        if in_str in hash_t:
            if hash_t[in_str] > 0:
                need_str -= 1
            hash_t[in_str] -= 1

        while need_str == 0:
            temp_str = s[left_point:i + 1]
            if len(temp_str) < lenth:
                min_str = temp_str
                lenth = len(temp_str)
            out_str = s[left_point]
            if out_str in hash_t:
                if hash_t[out_str] == 0:
                    need_str += 1
                hash_t[out_str] += 1
            left_point += 1

            if need_str > 0:
                break

    return min_str

print(cnt)

