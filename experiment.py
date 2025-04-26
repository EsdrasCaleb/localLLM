import os
import sys
import subprocess
import glob
import pandas as pd
import re
import lizard
import javalang

REQUIREMENTS_FILE = "requirements.txt"
LOCAL_MODELS_FILE = "models_local.txt"
WEB_MODELS_FILE = "models_web.txt"
PROJECTS_DIR = "../SF110"
BENCHMARK_URL = "http://www.evosuite.org/files/SF110-20130704-src.zip"

ENV_TEMPLATE = """apiKeys={api_key}
url={url}
model={model}
enableMultithreading=false
parentEnvPath=
phase=BENCHMARK
sleeptime={timeout}
max_tokens=1024
use_intention={intention}
onlyUpdateClass=true
max_prompt_tokens=3000
temperature={temp}
plugin={project}
timeout={timeout}
baseDir=../SF110/{index}_{project}
groupId={project_path}
chatunitest-tests=../SF110/{index}_{project}/chatunitest-tests_{model_name}/
benchmark_file=evosuitcsvs/evosuit_{project}.csv
artifactId={project}
compileSourceRoots=../SF110/{index}_{project}/src/main/java
buildPath=../SF110/{index}_{project}/target
artifactPath=../SF110/{index}_{project}/{project}.jar
classPaths=../SF110/lib/evosuite.jar:../SF110/lib/:/tmp/chatunitest-info/{project}/build/{project_path_dir}/{project}/data/:../SF110/{index}_{project}:../SF110/{index}_{project}/lib:../SF110/{index}_{project}/test-lib:../SF110/{index}_{project}/target:./src/main/resources/dependency
packaging=jar
"""

projects_chattester_mapping = {
    "1_tullibee":{
  "com.ib.client.EException": [],
  "com.ib.client.CodeMsgPair": [
    "code()",
    "msg()"
  ],
  "com.ib.client.Contract": [
    "clone()",
    "equals(Object)"
  ],
  "com.ib.client.ScannerSubscription": [
    "numberOfRows()",
    "instrument()",
    "locationCode()",
    "scanCode()",
    "abovePrice()",
    "belowPrice()",
    "aboveVolume()",
    "averageOptionVolumeAbove()",
    "marketCapAbove()",
    "marketCapBelow()",
    "moodyRatingAbove()",
    "moodyRatingBelow()",
    "spRatingAbove()",
    "spRatingBelow()",
    "maturityDateAbove()",
    "maturityDateBelow()",
    "couponRateAbove()",
    "couponRateBelow()",
    "excludeConvertible()",
    "scannerSettingPairs()",
    "stockTypeFilter()",
    "numberOfRows(int)",
    "instrument(String)",
    "locationCode(String)",
    "scanCode(String)",
    "abovePrice(double)",
    "belowPrice(double)",
    "aboveVolume(int)",
    "averageOptionVolumeAbove(int)",
    "marketCapAbove(double)",
    "marketCapBelow(double)",
    "moodyRatingAbove(String)",
    "moodyRatingBelow(String)",
    "spRatingAbove(String)",
    "spRatingBelow(String)",
    "maturityDateAbove(String)",
    "maturityDateBelow(String)",
    "couponRateAbove(double)",
    "couponRateBelow(double)",
    "excludeConvertible(String)",
    "scannerSettingPairs(String)",
    "stockTypeFilter(String)"
  ],
  "com.ib.client.EReader": [
    "run()",
    "stop()"
  ],
  "com.ib.client.EClientSocket": [
    "faMsgTypeName(int)",
    "serverVersion()",
    "TwsConnectionTime()",
    "eConnect(String, int, int)",
    "eDisconnect()",
    "cancelScannerSubscription(int)",
    "reqScannerParameters()",
    "reqScannerSubscription(int, ScannerSubscription)",
    "reqMktData(int, Contract, String, boolean)",
    "cancelHistoricalData(int)",
    "cancelRealTimeBars(int)",
    "reqHistoricalData(int, Contract, String, String, String, String, int, int)",
    "reqRealTimeBars(int, Contract, int, String, boolean)",
    "reqContractDetails(int, Contract)",
    "reqMktDepth(int, Contract, int)",
    "cancelMktData(int)",
    "cancelMktDepth(int)",
    "exerciseOptions(int, Contract, int, int, String, int)",
    "placeOrder(int, Contract, Order)",
    "reqAccountUpdates(boolean, String)",
    "reqExecutions(int, ExecutionFilter)",
    "cancelOrder(int)",
    "reqOpenOrders()",
    "reqIds(int)",
    "reqNewsBulletins(boolean)",
    "cancelNewsBulletins()",
    "reqAutoOpenOrders(boolean)",
    "reqAllOpenOrders()",
    "reqManagedAccts()",
    "requestFA(int)",
    "replaceFA(int, String)",
    "reqCurrentTime()",
    "reqFundamentalData(int, Contract, String)",
    "cancelFundamentalData(int)",
    "dataInputStream()"
  ],
  "com.ib.client.ContractDetails": [],
  "com.ib.client.OrderState": [
    "equals(Object)"
  ],
  "com.ib.client.Execution": [
    "equals(Object)"
  ],
  "com.ib.client.ExecutionFilter": [
    "equals(Object)"
  ],
  "com.ib.client.ComboLeg": [
    "equals(Object)"
  ],
  "com.ib.client.EClientErrors": [],
  "com.ib.client.EWrapperMsgGenerator": [
    "tickPrice(int, int, double, int)",
    "tickSize(int, int, int)",
    "tickOptionComputation(int, int, double, double, double, double)",
    "tickGeneric(int, int, double)",
    "tickString(int, int, String)",
    "tickEFP(int, int, double, String, double, int, String, double, double)",
    "orderStatus(int, String, int, int, double, int, int, double, int, String)",
    "openOrder(int, Contract, Order, OrderState)",
    "openOrderEnd()",
    "updateAccountValue(String, String, String, String)",
    "updatePortfolio(Contract, int, double, double, double, double, double, String)",
    "updateAccountTime(String)",
    "accountDownloadEnd(String)",
    "nextValidId(int)",
    "contractDetails(int, ContractDetails)",
    "contractMsg(Contract)",
    "bondContractDetails(int, ContractDetails)",
    "contractDetailsEnd(int)",
    "execDetails(int, Contract, Execution)",
    "execDetailsEnd(int)",
    "updateMktDepth(int, int, int, int, double, int)",
    "updateMktDepthL2(int, int, String, int, int, double, int)",
    "updateNewsBulletin(int, int, String, String)",
    "managedAccounts(String)",
    "receiveFA(int, String)",
    "historicalData(int, String, double, double, double, double, int, int, double, boolean)",
    "realtimeBar(int, long, double, double, double, double, long, double, int)",
    "scannerParameters(String)",
    "scannerData(int, int, ContractDetails, String, String, String, String)",
    "scannerDataEnd(int)",
    "currentTime(long)",
    "fundamentalData(int, String)",
    "deltaNeutralValidation(int, UnderComp)",
    "tickSnapshotEnd(int)"
  ],
  "com.ib.client.Util": [
    "StringIsEmpty(String)",
    "NormalizeString(String)",
    "StringCompare(String, String)",
    "StringCompareIgnCase(String, String)",
    "VectorEqualsUnordered(Vector, Vector)",
    "IntMaxString(int)",
    "DoubleMaxString(double)"
  ],
  "com.ib.client.TagValue": [
    "equals(Object)"
  ],
  "com.ib.client.TickType": [
    "getField(int)"
  ],
  "com.ib.client.UnderComp": [
    "equals(Object)"
  ],
  "com.ib.client.Order": [
    "equals(Object)"
  ],
  "com.ib.client.AnyWrapperMsgGenerator": [
    "error(Exception)",
    "error(String)",
    "error(int, int, String)",
    "connectionClosed()",
    "ioError(Exception)"
  ]
},
    "2_a4j":{
  "net.kencochrane.a4j.beans.Authors": [
    "getAuthor(int)",
    "toString()"
  ],
  "net.kencochrane.a4j.beans.ThirdPartyProductDetails": [
    "toString()"
  ],
  "net.kencochrane.a4j.beans.BrowseList": [
    "toString()"
  ],
  "net.kencochrane.a4j.beans.Directors": [
    "getDirector(int)",
    "toString()"
  ],
  "net.kencochrane.a4j.beans.RecentlyViewed": [
    "addProduct(MiniProduct)",
    "isInList(String)"
  ],
  "net.kencochrane.a4j.beans.SimilarProducts": [
    "getProduct(int)",
    "toString()"
  ],
  "net.kencochrane.a4j.util.LoadProperties": [
    "instance()"
  ],
  "net.kencochrane.a4j.beans.SellerProfileDetails": [
    "toString()"
  ],
  "net.kencochrane.a4j.util.a4jUtil": [
    "URLFriendlyName(String)",
    "stripString(String, String)",
    "getPrice(String)",
    "arrayToCommaString(ArrayList)",
    "encodeString(String)",
    "dencodeString(String)"
  ],
  "net.kencochrane.a4j.DAO.Search": [
    "Blended(String, String)",
    "Keyword(String, String, String, String)",
    "Generic(String, String, String, String, String, String)",
    "ActorSearch(String, String, String)",
    "ArtistSearch(String, String, String)",
    "AuthorSearch(String, String)",
    "DirectorSearch(String, String, String)",
    "ManufactureSearch(String, String, String)",
    "UpcSearch(String, String, String)",
    "ListmaniaSearch(String)",
    "WishListSearch(String)",
    "ThirdParty(String, String, String, String)",
    "SimilaritesSearch(String, String)"
  ],
  "net.kencochrane.a4j.beans.Reviews": [
    "toString()"
  ],
  "net.kencochrane.a4j.beans.MiniProduct": [
    "toString()"
  ],
  "net.kencochrane.a4j.beans.SellerProfile": [
    "toString()"
  ],
  "net.kencochrane.a4j.file.FileUtil": [
    "downloadOneASINFile(String, String, String, String, String)",
    "deleteFile(String)",
    "isAgeGood(File)",
    "renameFile(String, String)",
    "getASINFile(String, String, String, String)",
    "fetchASINFile(String, String, String, String)",
    "downloadBrowseNodeFile(String, String, String, String)",
    "getBrowseNodeFile(String, String, String)",
    "fetchBNFile(String, String, String)",
    "downloadBlendedSearchFile(String, String)",
    "downloadKeywordSearchFile(String, String, String, String)",
    "fetchBlendedSearchFile(String, String)",
    "fetchKeywordSearchFile(String, String, String, String)",
    "downloadGenericSearchFile(String, String, String, String, String, String)",
    "fetchGenericSearchFile(String, String, String, String, String, String)",
    "downloadThirdPartySearchFile(String, String, String, String)",
    "fetchThirdPartySearchFile(String, String, String, String)",
    "getAccessories(String, ArrayList)",
    "downloadAccessoriesFile(String, ArrayList, String)",
    "fetchAccessories(String, ArrayList)",
    "getSimilarItems(String, String)",
    "downloadSimilaritesFile(String, String, String)",
    "fetchSimilarItems(String, String)",
    "downloadCart(String)"
  ],
  "net.kencochrane.a4j.beans.Lists": [
    "getListId(int)",
    "toString()"
  ],
  "net.kencochrane.a4j.beans.FullProduct": [
    "addAccessory(MiniProduct)",
    "addSimilarItem(MiniProduct)",
    "printFullProduct()"
  ],
  "net.kencochrane.a4j.data.Query": [
    "queryGenerator(String, String, String, String, ArrayList)",
    "sendRequest(String)",
    "browseNodeQueryGenerator(String, String, String, String, String)",
    "BlendedSearchGenerator(String, String)",
    "KeywordSearchGenerator(String, String, String, String)",
    "SearchQueryGenerator(String, String, String, String, String, String)",
    "SearchThirdPartyGenerator(String, String, String, String)",
    "AddtoCart(String, String)",
    "AddToExistingCart(String, String, String, String)",
    "ClearCart(String, String)",
    "GetItemsFromCart(String, String)",
    "ModifyCart(String, String, String, String)",
    "RemoveFromCart(String, String, String)"
  ],
  "net.kencochrane.a4j.beans.BrowseNode": [
    "addSubNode(BrowseNode)",
    "getSubNode(String)",
    "printNode()",
    "toString()"
  ],
  "net.kencochrane.a4j.A4j": [
    "getFullProductFromASIN(String, String, String)",
    "BlendedSearch(String, String)",
    "KeywordSearch(String, String, String, String)",
    "ActorSearch(String, String, String)",
    "ArtistSearch(String, String, String)",
    "AuthorSearch(String, String)",
    "DirectorSearch(String, String, String)",
    "ManufactureSearch(String, String, String)",
    "UpcSearch(String, String, String)",
    "ListmaniaSearch(String)",
    "WishListSearch(String)",
    "ThirdParty(String, String, String, String)",
    "AddtoCart(String, String)",
    "addToExistingCart(String, String, String, String)",
    "clearCart(String, String)",
    "modifyCart(String, String, String, String)",
    "GetItemsFromCart(String, String)",
    "RemoveFromCart(String, String, String)"
  ],
  "net.kencochrane.a4j.beans.ListingProductInfo": [],
  "net.kencochrane.a4j.beans.ProductDetails": [
    "toString()"
  ],
  "net.kencochrane.a4j.beans.Artists": [
    "getArtist(int)",
    "toString()"
  ],
  "net.kencochrane.a4j.beans.Tracks": [
    "getTrack(int)",
    "toString()"
  ],
  "net.kencochrane.a4j.beans.Item": [
    "toString()"
  ],
  "net.kencochrane.a4j.beans.SellerSearch": [],
  "net.kencochrane.a4j.beans.CustomerReview": [
    "toString()"
  ],
  "net.kencochrane.a4j.beans.Items": [
    "toString()"
  ],
  "net.kencochrane.a4j.DAO.Cart": [
    "AddtoCart(String, String)",
    "addToExistingCart(String, String, String, String)",
    "clearCart(String, String)",
    "modifyCart(String, String, String, String)",
    "GetItemsFromCart(String, String)",
    "RemoveFromCart(String, String, String)"
  ],
  "net.kencochrane.a4j.beans.Mode": [],
  "net.kencochrane.a4j.beans.Starring": [
    "getActor(int)",
    "toString()"
  ],
  "net.kencochrane.a4j.beans.Platforms": [
    "getPlatform(int)",
    "toString()"
  ],
  "net.kencochrane.a4j.DAO.Product": [
    "getProduct(String, String, String)"
  ],
  "net.kencochrane.a4j.beans.ProductLine": [
    "toString()",
    "printProductList()"
  ],
  "net.kencochrane.a4j.beans.ShoppingCartResponse": [],
  "net.kencochrane.a4j.beans.BlendedSearch": [
    "toString()",
    "printProductList()"
  ],
  "net.kencochrane.a4j.beans.ModeList": [
    "addMode(Mode)",
    "getMode(String)"
  ],
  "net.kencochrane.a4j.beans.ShoppingCart": [
    "toString()",
    "getItem(String)"
  ],
  "net.kencochrane.a4j.beans.SellerFeedback": [
    "toString()"
  ],
  "net.kencochrane.a4j.beans.Features": [
    "getFeature(int)",
    "toString()"
  ],
  "net.kencochrane.a4j.beans.SellerSearchDetails": [],
  "net.kencochrane.a4j.beans.ListingProductDetails": [
    "toString()"
  ],
  "net.kencochrane.a4j.beans.Accessories": [
    "getAccessory(int)",
    "toString()"
  ],
  "net.kencochrane.a4j.beans.FeedBack": [
    "toString()"
  ],
  "net.kencochrane.a4j.beans.ProductInfo": [
    "toString()",
    "printProductList()"
  ],
  "net.kencochrane.a4j.beans.ThirdPartyProductInfo": [
    "toString()"
  ]
},
    "3_gaj":{
  "brain.ga.Genome": [
    "initialize()",
    "compareTo(Object)"
  ],
  "brain.ga.UniformCrossover": [
    "cross(Genome, Genome)"
  ],
  "brain.ga.GAEnumAllelesSet": [
    "allele()",
    "allele(int)",
    "size()"
  ],
  "brain.ga.SectMutator": [
    "mutate(Genome, double)"
  ],
  "brain.ga.VectorAllelesGenome": [],
  "brain.ga.RankSelector": [
    "select(Population)"
  ],
  "brain.ga.GAAlgorithm": [
    "evolve()"
  ],
  "brain.ga.Population": [
    "initialize(GAEnumAllelesSet)",
    "selectNextGenome()",
    "get(int)",
    "sort()"
  ],
  "brain.ga.VectorGenome": [
    "getGene(int)"
  ],
  "brain.ga.GAUtilities": [
    "flipCoin(double)",
    "nextPos(int)"
  ]
},
    "4_rif":{
  "com.densebrain.rif.server.RIFImplementationManager": [
    "registerImplementation(Class, Object)",
    "invoke(String, String, Object[])"
  ],
  "com.densebrain.rif.server.RIFService": [
    "invoke(String, String, String)"
  ],
  "com.densebrain.rif.client.RIFInvoker": [
    "invoke(String, Object[])"
  ],
  "com.densebrain.rif.server.transport.WebServiceContainer": [
    "newInstance(String, int, String)",
    "newInstance(ConfigurationContext)",
    "configureService(Class, String, String)",
    "configureService(WebServiceDescriptor)",
    "restartContainer()",
    "startContainer()",
    "stopContainer()",
    "getEPRForService(String, String)"
  ],
  "com.densebrain.rif.client.service.types.Factory": [
    "parse(javax.xml.stream.XMLStreamReader)"
  ],
  "com.densebrain.rif.client.RIFManagerFactory": [
    "getManager(String)",
    "getInvoker(String, Class)",
    "getImpl(String, Class)"
  ],
  "com.densebrain.rif.client.service.types.InvokeResponse": [
    "getPullParser(javax.xml.namespace.QName)"
  ],
  "com.densebrain.rif.client.RIFClassLoader": [],
  "com.densebrain.rif.server.transport.WebServiceDescriptor": [
    "hashCode()",
    "equals(Object)"
  ],
  "com.densebrain.rif.server.test.TestWS": [],
  "com.densebrain.rif.client.RIFManager": [
    "getInvoker(Class)"
  ],
  "com.densebrain.rif.client.service.RIFServiceStub": [
    "invoke(com.densebrain.rif.client.service.types.Invoke)"
  ],
  "com.densebrain.rif.client.service.types.Invoke": [
    "getPullParser(javax.xml.namespace.QName)"
  ],
  "com.densebrain.rif.server.RIFServer": [
    "start()",
    "stop()"
  ],
  "com.densebrain.rif.util.ObjectUtility": [
    "serializeObject(Object)",
    "encodeBytes(byte[])",
    "deserializeObjectBase64Encoded(String)",
    "deserializeObject(byte[])",
    "decodeString(String)"
  ]
},
  "5_templateit":{
  "org.apache.poi.hssf.usermodel.HSSFDataFormat": [
    "getBuiltinFormat(String)",
    "getFormat(String)",
    "getFormat(short)",
    "getBuiltinFormat(short)"
  ],
  "org.templateit.DynamicTemplate": [
    "height()",
    "width()",
    "absoluteReference(int, int)",
    "getRowHeight(int)",
    "getCell(int, int)"
  ],
  "org.templateit.util.DelimitedFileReader": [
    "hasNext()",
    "next()",
    "remove()"
  ],
  "org.templateit.TemplateIt": [
    "main(String[])"
  ],
  "org.templateit.util.FormulaUtil": [
    "offsetRelativeReferences(HSSFWorkbook, String, int, int)"
  ],
  "org.templateit.Poi2ItextUtil": [
    "colorPOI2Itext(HSSFColor)",
    "copyBackgroundColor(HSSFCell, PdfPCell)",
    "copyCellHorisontalAlignment(HSSFCell, PdfPCell)",
    "copyCellBorders(HSSFCell, PdfPCell)",
    "resetRightBorder(HSSFCell, PdfPCell)",
    "chooseFont(HSSFFont)",
    "chooseFontFamily(HSSFFont, int)",
    "chooseFont(short)"
  ],
  "org.templateit.PdfWriter": [
    "writePdf(OutputStream)"
  ],
  "org.templateit.TemplateProcessor": [
    "process(Iterator, File)",
    "process(Iterator, OutputStream)",
    "keepSheet(String)",
    "generateNewSheet(String, String, Iterator)"
  ],
  "org.templateit.MergeData": [
    "collectMergeData()",
    "getMergeRegionAt(int, int)"
  ]
},
    "6_jnfe":{
  "br.com.jnfe.base.service.NFeCalculatorImpl": [
    "calculate(ICMS)",
    "calculate(ICMSST)",
    "calculate(ICMSExt)",
    "calculate(IPI)",
    "calculate(PIS)",
    "calculate(COFINS)"
  ],
  "br.com.jnfe.base.service.LoggingFaultMessageResolver": [
    "resolveFault(WebServiceMessage)"
  ],
  "br.com.jnfe.base.TUFs": [],
  "br.com.jnfe.base.service.DOMNFeKeyInfoBuilder": [
    "newKeyInfo(Certificate)"
  ],
  "br.com.jnfe.base.TransportKeyStoreBean": [
    "afterPropertiesSet()",
    "toString()",
    "openTransportStore()",
    "openTransportKeyManagerFactory()"
  ],
  "br.com.jnfe.base.DefaultNamespacePrefixMapper": [
    "getPreferredPrefix(String, String, boolean)"
  ],
  "br.com.jnfe.base.service.DOMNFeFileReader": [
    "loadAndSign(String, String)",
    "loadAndSign(InputStream, String)"
  ],
  "br.com.jnfe.base.service.DOMNFeSigantureFactoryBean": [
    "afterPropertiesSet()"
  ],
  "br.com.jnfe.base.pl006.RequestAdapterImpl": [
    "newCabec()",
    "newRequest(String, String, String)"
  ],
  "br.com.jnfe.base.pl005d.RequestAdapterImpl": [
    "newCabec()",
    "newRequest(String, String, String)"
  ],
  "br.com.jnfe.base.service.DOMNFeSignatureHandler": [
    "sign(XMLStructure, String)"
  ],
  "br.com.jnfe.base.service.Pkcs12SecurityHandlerBean": [
    "handle(Element, Element, SecurityCallBack)",
    "afterPropertiesSet()",
    "loadKeyStore()"
  ],
  "br.com.jnfe.base.service.DOMNFeSignatureBuilder": [
    "afterPropertiesSet()",
    "build(Element, Element, Certificate, PrivateKey)"
  ],
  "br.com.jnfe.base.service.SimpleSecurityHandlerBean": [
    "handle(Element, Element, SecurityCallBack)",
    "afterPropertiesSet()"
  ],
  "br.com.jnfe.base.util.SecurityUtils": [
    "openStore(String, Resource, char[])",
    "openStore(Resource, char[])",
    "openStore(String, String, char[])",
    "openStore(String, char[])",
    "openTrustStore(char[])",
    "openTrustStore(String, char[])",
    "installCertificate(String, String)",
    "installCertificate(String, String, String)",
    "main(String[])"
  ],
  "br.com.jnfe.base.TProvince": [
    "toString()"
  ],
  "br.com.jnfe.base.ConsReciNFe": []
},
"7_sfmis":{
  "com.hf.sfm.util.BasePara": [
    "single2plannar()"
  ],
  "com.hf.sfm.system.business.Login": [
    "destroy()",
    "doGet(HttpServletRequest, HttpServletResponse)",
    "doPost(HttpServletRequest, HttpServletResponse)",
    "init()"
  ],
  "com.hf.sfm.sfmis.personinfo.business.PersonInfoMgr": [
    "saveOrUpdate(APersonInfo)",
    "deleteByIds(String[])"
  ],
  "com.hf.sfm.util.HibernateSessionFactory": [
    "currentSession()",
    "closeSession()",
    "main(String[])"
  ],
  "com.hf.sfm.system.pdo.AWorker": [],
  "com.hf.sfm.system.pdo.Menu": [],
  "com.hf.sfm.crypt.Base64": [
    "altBase64ToByteArray(String)",
    "base64ToByteArray(String)",
    "byteArrayToAltBase64(byte[])",
    "byteArrayToBase64(byte[])",
    "main(String[])"
  ],
  "com.hf.sfm.util.DataSource": [
    "getPlanarArrData(BasePara)",
    "getGridData(BasePara)",
    "getComboData(BasePara)",
    "getSession(HttpSession, String)",
    "main(String[])"
  ],
  "com.hf.sfm.sfmis.personinfo.pdo.APersonInfo": [],
  "com.hf.sfm.system.business.MenuManage": [
    "saveOrUpdate(Menu)",
    "del(String[])"
  ],
  "com.hf.sfm.system.pdo.AGroup": [],
  "com.hf.sfm.util.DaoFactory": [
    "currentSession()",
    "closeSession()",
    "commit()",
    "beginTransaction()",
    "rollback()",
    "encrypt(String)",
    "decrypt(String)",
    "save(Object)",
    "update(Object)",
    "closeAll()"
  ],
  "com.hf.sfm.sfmis.department.pdo.ADepartment": [],
  "com.hf.sfm.system.business.WorkerMgr": [
    "saveOrUpdate(AWorker)",
    "deleteByIds(String[])"
  ],
  "com.hf.sfm.util.ListRange": [],
  "com.hf.sfm.util.OddParamsOfArrayInLoader": [],
  "com.hf.sfm.util.Loader": [
    "run(BasePara)",
    "parseXML()",
    "loadDataWithSql()",
    "getParams(Query, String[][])",
    "collectToMap(String)",
    "collectToMap()"
  ],
  "com.hf.sfm.filter.setCharacterEncodingFilter": [
    "destroy()",
    "doFilter(ServletRequest, ServletResponse, FilterChain)",
    "init(FilterConfig)"
  ]
}
}

def load_or_create_env(env_path=".env"):
  if not os.path.exists(env_path):
    with open(env_path, 'w') as f:
      f.write("# .env file created\n")


  with open(env_path, 'r') as f:
    for line in f:
      # Remove leading/trailing whitespace and newline characters
      line = line.strip()

      # Ignore comments and empty lines
      if line and not line.startswith("#"):
        key_value = line.split("=", 1)

        # Ensure there are exactly two parts: key and value
        if len(key_value) == 2:
          key, value = key_value
          key = key.strip()
          value = value.strip()

          # Optionally, interpret booleans and numbers
          if value.lower() in ["true", "false"]:
            value = value.lower() == "true"
          elif value.isdigit():
            value = int(value)

          env_dict[key] = value

    return env_dict

env_dict = load_or_create_env()

def add_env_variable(key, value, env_path=".env"):
  env_dict[key] = value
  if not os.path.exists(env_path):
    load_or_create_env(env_path)

  lines = []
  found = False

  with open(env_path, 'r') as f:
    lines = f.readlines()

  for i, line in enumerate(lines):
    if line.strip().startswith(f"{key}="):
      lines[i] = f"{key}={value}\n"
      found = True
      break

  if not found:
    lines.append(f"{key}={value}\n")

  with open(env_path, 'w') as f:
    f.writelines(lines)

def run_test_smell_detector(input_csv_path,  jar_name="TestSmellDetector.jar"):
  # Chama o JAR com o arquivo CSV de entrada
  subprocess.run(["java", "-jar", jar_name, input_csv_path], check=True)


  # Procura o CSV de saída gerado dentro de tsDetect Output_TestSmellDetection_
  output_files = glob.glob(os.path.join("Output_TestSmellDetection_*.csv"))
  if not output_files:
    raise FileNotFoundError("Nenhum arquivo CSV foi gerado pelo TestSmellDetector.")

  output_csv = output_files[0]

  # Lê o CSV em um DataFrame
  df = pd.read_csv(output_csv)

  # Apaga o arquivo CSV gerado
  os.remove(output_csv)

  return df

def count_unique_methods_tested(evosuite_file, source_file):
  if not os.path.exists(evosuite_file) or not os.path.exists(source_file):
    return 0

  try:
    with open(source_file, 'r', encoding='utf-8') as f:
      sut_code = f.read()
  except UnicodeDecodeError:
    with open(source_file, 'r', encoding='latin1') as f:
      sut_code = f.read()

  try:
    sut_tree = javalang.parse.parse(sut_code)
  except:
    return 0

  sut_methods = set()
  for _, node in sut_tree.filter(javalang.tree.MethodDeclaration):
    sut_methods.add(node.name)

  try:
    with open(evosuite_file, 'r', encoding='utf-8') as f:
      test_code = f.read()
  except UnicodeDecodeError:
    with open(evosuite_file, 'r', encoding='latin1') as f:
      test_code = f.read()

  methods_tested = {m for m in sut_methods if f".{m}(" in test_code}
  return len(methods_tested)

def find_existing_evosuite_tests(projects_chattester_mapping, projects_dir):
    result = {}
    data = []
    for project, classes in projects_chattester_mapping.items():
        project_path = os.path.join(projects_dir, project, "evosuite-tests")
        existing_files = []
        project_name = project.split("_")[1]

        for class_path in classes:
            relative_path = os.path.join(*class_path.split('.')) + "EvoSuiteTest.java"
            evo_path = os.path.join(project_path, relative_path)

            if os.path.isfile(evo_path):
                sut_relative = os.path.join("src", "main", "java", *class_path.split('.')) + ".java"
                sut_path = os.path.join(projects_dir, project, sut_relative)
                if(os.path.isfile(sut_path)):
                  if(not evo_path in existing_files):
                    existing_files.append(evo_path)
                    data.append([project_name, evo_path, sut_path])
                else:
                  print("error cant find destination class to "+evo_path)

        if existing_files:
            result[project_name] = existing_files

    return result,data


def merge_test_data(final_dt, smell_dt):
  # Faz o merge usando 'file' de df1 e 'TestFilePath' de df2
  merged_df = final_dt.merge(
    smell_dt,
    left_on="file",
    right_on="TestFilePath",
    how="inner"
  )

  # Seleciona e reordena as colunas conforme especificado
  final_columns = [
    "project", "file", "num_interactions", "num_corrections", "result", "model", "test_number",
    "mutation_null", "mutation_var", "mutation_bool", "mutation_aritime", "mutation_logic", "mutation_relat",
    "number_of_sut_methods","number_of_tests","NumberOfMethods", "Assertion Roulette", "Conditional Test Logic",
    "Constructor Initialization","Default Test", "EmptyTest", "Exception Catching Throwing", "General Fixture",
    "Mystery Guest", "Print Statement","Redundant Assertion", "Sensitive Equality", "Verbose Test", "Sleepy Test",
    "Eager Test", "Lazy Test","Duplicate Assert", "Unknown Test", "IgnoredTest", "Resource Optimism",
    "Magic Number Test", "Dependent Test"
  ]

  return merged_df[final_columns]


def generate_dt_from_evosuite_files(existing_files_by_project):
  data = []

  # Montar todos os caminhos sut_paths_by_project
  sut_paths_by_project = {}
  for project in existing_files_by_project.keys():
    filename = f"evosuittestsemll_{project}"
    if os.path.exists(filename):
      df_sut = pd.read_csv(filename, header=None, names=["project", "file", "source_file"])
      for _, row in df_sut.iterrows():
        sut_paths_by_project[row["file"]] = row["source_file"]
    else:
      print(f"Arquivo {filename} não encontrado para o projeto {project}")

  for project, files in existing_files_by_project.items():
    for file_path in files:
      try:
        with open(file_path, 'r', encoding='utf-8') as f:
          content = f.read()
        num_tests = len(re.findall(r'@Test\b', content))
      except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        num_tests = -1

      source_file = sut_paths_by_project.get(file_path, None)
      if source_file:
        num_sut_methods = count_unique_methods_tested(file_path, source_file)
      else:
        num_sut_methods = 0

      data.append([
        project,  # project
        file_path,  # file
        1,  # num_interactions
        0,  # num_corrections
        "SUCCESS",  # result
        "evosuite",  # model
        0,  # test_number
        -1, -1, -1, -1, -1, -1,  # mutation_* colunas
        num_sut_methods,
        num_tests
      ])

  df = pd.DataFrame(data, columns=[
    "project", "file", "num_interactions", "num_corrections", "result", "model",
    "test_number", "mutation_null", "mutation_var", "mutation_bool",
    "mutation_aritime", "mutation_logic", "mutation_relat",
    "number_of_sut_methods", "number_of_tests"
  ])

  return df

def analyze_code_metrics(file_path):
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

  analysis = lizard.analyze_file(file_path)
  return analysis.nloc, analysis.CCN, analysis.token_count, len(analysis.function_list)


def count_assertions_in_methods(java_file):
  if not os.path.exists(java_file):
    raise FileNotFoundError(f"Arquivo não encontrado: {java_file}")

  with open(java_file, 'r', encoding='utf-8') as file:
    code = file.read()
  tree = javalang.parse.parse(code)

  total_assertions = 0
  methods_without_assertions = 0
  total_methods = 0

  for _, node in tree.filter(javalang.tree.MethodDeclaration):
    total_methods += 1
    assertion_count = sum(1 for _, stmt in node.filter(javalang.tree.MethodInvocation) if
                          (stmt.member.startswith("assert") or stmt.member.startswith("fail")))

    total_assertions += assertion_count
    if assertion_count == 0:
      methods_without_assertions += 1

  return total_assertions, methods_without_assertions, total_methods

def check_requirements():
    try:
        with open(REQUIREMENTS_FILE, "r") as file:
            packages = file.read().splitlines()

        missing_packages = []
        for package in packages:
            try:
                subprocess.check_output([sys.executable, "-m", "pip", "show", package.split("==")[0]])
            except subprocess.CalledProcessError:
                missing_packages.append(package)

        if missing_packages:
            print(f"Missing packages: {', '.join(missing_packages)}")
            install = input("Do you want to install them? (y/n): ").strip().lower()
            if install == "y":
                subprocess.run([sys.executable, "-m", "pip", "install", *missing_packages])
            else:
                sys.exit("Required packages are missing. Exiting.")

    except FileNotFoundError:
        sys.exit(f"Error: {REQUIREMENTS_FILE} not found.")


def check_benchmark():
    if not os.path.exists(PROJECTS_DIR):
        print(f"\nBenchmark not found at {PROJECTS_DIR}.")
        print(f"Please download it from {BENCHMARK_URL}")
        print(f"Extract it to the root folder and rename it to 'SF110'.\n")
        input("Press Enter after completing this step...")
        if not os.path.exists(PROJECTS_DIR):
            sys.exit("SF110 directory not found. Exiting.")

def select_option():
    print("\nChoose an option:")
    print("a - Run a single model benchmark")
    print("b - Run all benchmarks")
    print("c - Generate EvoSuite benchmark")
    print("d - Fuse all generated data in one file")

    choice = input("Enter your choice (a/b/c): ").strip().lower()
    if choice not in ("a", "b", "c","d"):
        sys.exit("Invalid choice. Exiting.")
    return choice


def get_models():
    model_type = input("Do you want to use a local or web model? (local/web): ").strip().lower()
    if model_type not in ("local", "web"):
        sys.exit("Invalid choice. Exiting.")

    models_file = LOCAL_MODELS_FILE if model_type == "local" else WEB_MODELS_FILE
    try:
        with open(models_file, "r") as file:
            models = file.read().splitlines()
            if not models:
                sys.exit(f"No models found in {models_file}. Exiting.")
    except FileNotFoundError:
        sys.exit(f"Error: {models_file} not found.")


    print("\nAvailable models:")
    for i, model in enumerate(models, start=1):
        print(f"{i}. {model}")

    model_index = int(input("Select a model number: ")) - 1
    if model_index < 0 or model_index >= len(models):
        sys.exit("Invalid model choice. Exiting.")

    selected_model = models[model_index]

    api_key = ""
    if model_type == "web":
      if((model_index==0 or model_index==5) and "g_tokens" not in env_dict):
        print("You don't have Google Gemini API keys configured.")
        print("Generate your Gemini API keys here:")
        print("  - https://aistudio.google.com/app/apikey")
        keys = input("Paste your Gemini API keys here, separated by commas if you have more than one: ").strip()
        add_env_variable("g_tokens", keys)
      if (model_index == 1 or model_index == 4):
        if("MISTRAL_API_KEY" not in env_dict):
          print("You don't have a Mistral API key configured.")
          print("Generate your Mistral API key here:")
          print("  - https://console.mistral.ai/")
          key = input("Paste your Mistral API key here, separated by commas if you have more than one: ").strip()
          add_env_variable("MISTRAL_API_KEY", key)
        api_key = env_dict["MISTRAL_API_KEY"]
      if (model_index == 2):
        if "gpt_key" not in env_dict:
          print("You don't have an OpenAI GPT API key configured.")
          print("Generate your OpenAI API key here:")
          print("  - https://platform.openai.com/account/api-keys")
          key = input("Paste your OpenAI API key here, separated by commas if you have more than one: ").strip()
          add_env_variable("gpt_key", key)
        api_key = env_dict["gpt_key"]
      if (model_index == 6):
        if "CHUTES_API_KEY" not in env_dict:
          print("You don't have a Chutes.ai API key configured.")
          print("Generate your Chutes.ai API key here:")
          print("  - https://chutes.ai/app/api")
          key = input("Paste your Chutes.ai API key here, separated by commas if you have more than one: ").strip()
          add_env_variable("CHUTES_API_KEY", key)
          env_dict["CHUTES_API_KEY"] = key
    else:
      if "HF_TOKEN" not in env_dict:
        print("You don't have a HuggingFace access token configured.")
        print("Generate one here: https://huggingface.co/settings/tokens")
        token = input("Paste your HuggingFace token here: ").strip()
        add_env_variable("HF_TOKEN", token)

    return selected_model, api_key


def get_projects():
    try:
        all_dirs = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

        # Filter projects that start with a number followed by underscore
        numbered_projects = []
        for project in all_dirs:
            parts = project.split('_', 1)
            if len(parts) == 2:
                try:
                    number = int(parts[0])
                    numbered_projects.append((number, project))
                except ValueError:
                    continue

        # Sort projects by number
        numbered_projects.sort()

        if not numbered_projects:
            sys.exit(f"No numbered projects found in {PROJECTS_DIR}. Exiting.")
    except FileNotFoundError:
        sys.exit(f"Error: {PROJECTS_DIR} not found.")

    # Filter to show only projects 1-7
    projects_1_to_7 = []
    print("\nAvailable projects:")
    for number, project in numbered_projects:
        if number < 7:
            projects_1_to_7.append(project)
            print(f"{number}. {project}")

    if not projects_1_to_7:
        sys.exit("No projects with numbers 1-7 found. Exiting.")

    selected_projects = input("Enter project numbers separated by commas: ").strip()
    selected_numbers = [int(i) for i in selected_projects.split(",")]

    # Validate selected numbers
    for num in selected_numbers:
        if num < 1 or num > 7:
            sys.exit("Invalid project selection. Exiting.")

    # Return the selected projects
    return [proj for num, proj in numbered_projects if num in selected_numbers]


def generate_env_file(project, model, api_key):
    env_content = ENV_TEMPLATE.format(
        api_key=api_key or "XXXKEYXXX",
        url="https://example.com",  # Modify as needed
        model=model,
        timeout=30,
        intention="true",
        temp=0.7,
        project=project,
        index=1,  # Modify as needed
        project_path=project,
        project_path_dir=project,
        model_name=model
    )

    env_filename = f"{project}.env"
    with open(env_filename, "w") as file:
        file.write(env_content)

    print(f"Generated {env_filename}")

env_data =load_or_create_env(env_path=".env")

def get_model_projects():
  model, api_key = get_models()
  project = get_projects()
  generate_env_file(project, model, api_key)
  return project, model


def main():
    check_requirements()
    check_benchmark()
    if len(sys.argv) > 1:
      option = sys.argv[1]
    else:
      option = select_option()
    match(option):
      case "a":
        projects, model = get_model_projects()

      case "b":
        projects, model = get_model_projects()
      case "c":
        evosuite_data,smell_evo_data = find_existing_evosuite_tests(projects_chattester_mapping,PROJECTS_DIR)
        df = pd.DataFrame(smell_evo_data)
        #pd.set_option('display.max_colwidth', None)  # mostra conteúdo completo das colunas
        df.to_csv("evotssmell.csv", index=False, header=False)
        ts_df = run_test_smell_detector("evotssmell.csv")
        finaldt = generate_dt_from_evosuite_files(evosuite_data)
        finaldt = merge_test_data(finaldt,ts_df)
        finaldt[['lizard_nloc', 'lizard_ccn', 'lizard_token', 'lizard_function_count']] = finaldt['file'].apply(
          lambda x: pd.Series(analyze_code_metrics(x)))
        finaldt[['total_assertion', 'methods_without_assertions', 'total_methods']] = finaldt['file'].apply(
          lambda x: pd.Series(count_assertions_in_methods(x)))
        finaldt.to_csv("evosuite_final.csv", index=False)
        print("Evosuite files in 'evosuite_final.csv'")
      case "d":
        print("All files merged into finaldata.csv")


if __name__ == "__main__":
    main()
