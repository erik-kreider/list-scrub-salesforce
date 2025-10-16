account_sql_query = '''
SELECT
    a.id as AccountID,
    a.name as company,
    a.Website,
    a.phone as Phone,
    a.primary_line_of_business__c as LOB,
    a.team__c as team,
    a.ShippingStreet,
    a.ShippingCity,
    a.ShippingState,
    a.ShippingPostalCode
FROM dbo.Account a
WHERE a.account_status__c <> 'internal'
  AND a.SourceSystem__c = 'Provider'

UNION ALL

SELECT
    a.id as AccountID,
    a.dba__c as company,
    a.Website,
    a.phone as Phone,
    a.primary_line_of_business__c as LOB,
    a.team__c as team,
    a.ShippingStreet,
    a.ShippingCity,
    a.ShippingState,
    a.ShippingPostalCode
FROM dbo.Account a
WHERE a.account_status__c <> 'internal'
  AND a.dba__c IS NOT NULL
  AND a.SourceSystem__c = 'Provider'

UNION ALL

SELECT
    a.id as AccountID,
    substring(a.name, charindex('fka', a.name) + 4, 100) as company,
    a.Website,
    a.phone as Phone,
    a.primary_line_of_business__c as LOB,
    a.team__c as team,
    a.ShippingStreet,
    a.ShippingCity,
    a.ShippingState,
    a.ShippingPostalCode
FROM dbo.Account a
WHERE a.account_status__c <> 'internal'
  AND a.name LIKE '%fka%'
  AND a.SourceSystem__c = 'Provider'

UNION ALL

SELECT
    a.id as AccountID,
    substring(a.name, charindex('dba', a.name) + 4, 100) as company,
    a.Website,
    a.phone as Phone,
    a.primary_line_of_business__c as LOB,
    a.team__c as team,
    a.ShippingStreet,
    a.ShippingCity,
    a.ShippingState,
    a.ShippingPostalCode
FROM dbo.Account a
WHERE a.account_status__c <> 'internal'
  AND a.name LIKE '%dba%'
  AND a.SourceSystem__c = 'Provider'
'''

contact_sql_query = '''
SELECT
    a.id as AccountID,
    c.id as ContactID,
    c.FirstName,
    c.LastName,
    isnull(c.phone, a.phone) as Phone,
    c.Title,
    c.Email,
    Case when c.contact_status__c = 'Active' then 1 else 0 end as 'Status'
FROM SalesforceOneReplicated.dbo.Account a
LEFT JOIN SalesforceOneReplicated.dbo.contact c
    ON a.id = c.AccountId
WHERE c.id IS NOT NULL
  AND a.SourceSystem__c = 'Provider'
'''