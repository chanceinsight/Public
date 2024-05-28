USE ContosoRetailDW;

WITH FirstTransactionDates AS (
    SELECT customerkey,MIN([DateKey]) AS first_transaction_date
    FROM [dbo].[FactOnlineSales]
    GROUP BY customerkey
)

SELECT 
cust.CustomerKey as CustomerKey, 
DATEDIFF(DAY, [BirthDate], GETDATE())/365 as Age, 
CASE 
WHEN [gender] = 'm' THEN 0
WHEN [gender] = 'f' THEN 1
END as Gender,
cust.NumberCarsOwned as NumberCarsOwned,
cust.HouseOwnerFlag as HouseOwnerFlag,
cust.YearlyIncome as YearlyIncome, 
a.FirstPurchaseAmount as FirstPurchaseAmount,
b.first12months_salesamount as LifetimeSales

FROM 
dbo.DimCustomer cust

--Build FirstPurchaseAmount table a
LEFT OUTER JOIN
(
SELECT CustomerKey, SalesAmount as FirstPurchaseAmount
FROM
(
SELECT 
[CustomerKey] as CustomerKey, 
[SalesOrderNumber] as SalesOrderNumber,
ROW_NUMBER() OVER (PARTITION BY [CustomerKey] ORDER BY [SalesOrderNumber] ASC) AS OrderNumber,
SUM([SalesAmount]) as SalesAmount
FROM
[dbo].[FactOnlineSales]
GROUP BY [CustomerKey], [SalesOrderNumber] 
) as y
WHERE OrderNumber = 1
) a
ON cust.CustomerKey = a.CustomerKey
LEFT OUTER JOIN

-- Build Sales and 12 Month Orders table b
(
SELECT 
    f.customerkey,
    SUM(CASE WHEN DATEDIFF(month, ft.first_transaction_date, f.DateKey) < 12 THEN f.salesamount 
	ELSE 0 
	END) AS first12months_salesamount

FROM [dbo].[FactOnlineSales] f
LEFT OUTER JOIN FirstTransactionDates ft ON 
f.customerkey = ft.customerkey
GROUP BY f.CustomerKey
) as b
ON cust.CustomerKey = b.CustomerKey

LEFT OUTER JOIN

--Build FirstOrderDate Table c
(
SELECT CustomerKey, DateKey as FirstOrderDate
FROM
(
SELECT 
s.CustomerKey as CustomerKey, 
[DateKey] as DateKey, 
ROW_NUMBER() OVER (PARTITION BY [CustomerKey] ORDER BY [DateKey] ASC) AS OrderNumber
FROM
[dbo].[FactOnlineSales] s
) as x
WHERE OrderNumber = 1
)  as g
ON cust.CustomerKey = g.CustomerKey


WHERE cust.CustomerType = 'Person'
