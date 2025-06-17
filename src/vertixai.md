



askPdf % gcloud ai indexes create --display-name="pdf-search-index" --region=us-central1 --metadata-file=index_config.json

gcloud ai operations describe 3807631684933779456 --index=8454630890109140992 --region=us-central1 --project=aiproject-455602

Index details:
Name: pdf-search-index
Index resource: projects/129285674200/locations/us-central1/indexes/8454630890109140992
Dimensions: 384
Distance measure: DOT_PRODUCT_DISTANCE

deploy the endx

gcloud ai index-endpoints deploy-index 6246424513441955840 --index=8454630890109140992 --region=us-central1 --project=aiproject-455602 --deployed-index-id=pdf_search_deployed_index --display-name="PDF Search Deployed Index"


check the status 

metadata:
  '@type': type.googleapis.com/google.cloud.aiplatform.v1.DeployIndexOperationMetadata
  deployedIndexId: pdf_search_deployed_index
  genericMetadata:
    createTime: '2025-06-17T00:56:35.652267Z'
    updateTime: '2025-06-17T00:56:35.652267Z'
name: projects/129285674200/locations/us-central1/indexEndpoints/6246424513441955840/operations/2262193325303922688




gcloud ai operations describe 2262193325303922688 --index-endpoint=6246424513441955840 --region=us-central1 --project=aiproject-455602




Index resource ID: 930241812679884800
Operation ID: 4211407539025215488

