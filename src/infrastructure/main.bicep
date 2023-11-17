// Core parameters:
param location string = resourceGroup().location


param appName string ='saladyolodev'

// Storage parameters:
param StorageAccountName string = 'sasaladyolodev'
param inputContainerName string = 'requests'
param outputContainerName string= 'yolo-results'
param failsContainerName string = 'failed-requests'
param RequestsQueueName string = 'requestsqueue'

// Storage vars:
var storageAccountSkuName = 'Standard_LRS'

// Event grid vars:
var SystemTopicName = '${appName}-topic'
var RequestsSubscriptionName = '${appName}-${inputContainerName}'


resource StorageAccount 'Microsoft.Storage/storageAccounts@2021-09-01' = {
  name: StorageAccountName
  location: location
  sku: {
    name: storageAccountSkuName
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    publicNetworkAccess: 'Enabled'
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Allow'
      ipRules: []
      virtualNetworkRules: []
    }
    encryption: {
      keySource: 'Microsoft.Storage'
      services: {
        blob: {
          enabled: true
        }
      }
    }
  }
}

resource BlobServices 'Microsoft.Storage/storageAccounts/blobServices@2021-04-01' = {
  parent: StorageAccount
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: false
      days: 7
    }
  }
}

var Containers = [
  {
    containerName: inputContainerName
    publicAccess: 'None'
  }
  {
    containerName: outputContainerName
    publicAccess: 'None'
  }
  {
    containerName: failsContainerName
    publicAccess: 'None'
  }
  
]

resource blobContainers 'Microsoft.Storage/storageAccounts/blobServices/containers@2021-08-01' = [for container in Containers: {
  name: container.containerName
  parent: BlobServices
  properties: {
    publicAccess: container.publicAccess
  }
}]


var Queues = [
  {
    queueName: RequestsQueueName
  }
  {
    queueName: 'poison-${RequestsQueueName}'
  }
]

resource QueueServices 'Microsoft.Storage/storageAccounts/queueServices@2021-09-01' = if (!empty(Queues)) {
  parent: StorageAccount
  name: 'default'
  properties: {}
}


resource StorageQueues 'Microsoft.Storage/storageAccounts/queueServices/queues@2021-09-01' = [for queue in Queues: {
  parent: QueueServices
  name: queue.queueName
  properties: {}
}]

resource Topic 'Microsoft.EventGrid/systemTopics@2022-06-15' = {
  name: SystemTopicName
  location: location
  properties: {
    source: StorageAccount.id
    topicType: 'Microsoft.Storage.StorageAccounts'
  }
  dependsOn: [
    StorageAccount
  ]
}

resource RequestsSubscription 'Microsoft.EventGrid/systemTopics/eventSubscriptions@2022-06-15'= {
  name: RequestsSubscriptionName
  parent: Topic
  properties: {
    destination: {
      endpointType: 'StorageQueue'
      properties: {
        queueName: RequestsQueueName
        resourceId: StorageAccount.id
      }  
    }
    eventDeliverySchema: 'EventGridSchema'
    filter: {
      enableAdvancedFilteringOnArrays: false
      includedEventTypes: [
        'Microsoft.Storage.BlobCreated'
      ]
      isSubjectCaseSensitive: false
      subjectBeginsWith: '/blobServices/default/containers/${inputContainerName}'
      subjectEndsWith: 'txt'
    }
    retryPolicy: {
      eventTimeToLiveInMinutes: 1440
      maxDeliveryAttempts: 30
    }
  }
}

