package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Properties;

public class Query_browseNodeQueryGenerator_3_0_Test {

    @Test
    public void browseNodeQueryGeneratorTest() throws Exception {
        Query query = new Query();
        query.serverURL = "http://example.com";
        query.associatesID = "1234567890";
        query.token = "DSB0XDDW1GQ3S";
        query.searchType = "searchType";
        query.type = "type";
        query.page = "1";
        query.offer = "offer";
        query.searchValues = new ArrayList();
        Field field = Query.class.getDeclaredField("searchType");
        field.setAccessible(true);
        field.set(query, "newSearchType");
        Field field2 = Query.class.getDeclaredField("type");
        field2.setAccessible(true);
        field2.set(query, "newType");
        Field field3 = Query.class.getDeclaredField("page");
        field3.setAccessible(true);
        field3.set(query, "2");
        Field field4 = Query.class.getDeclaredField("offer");
        field4.setAccessible(true);
        field4.set(query, "newOffer");
        Field field5 = Query.class.getDeclaredField("searchValues");
        field5.setAccessible(true);
        query.searchValues.add("searchValue1");
        query.searchValues.add("searchValue2");
        String result = query.browseNodeQueryGenerator("type", "1", "offer", "mode", "browseNode");
        String expected = "http://example.com?t=1234567890&dev-t=DSB0XDDW1GQ3S&BrowseNodeSearch=browseNode&mode=mode&type=type&page=1&offer=newOffer&f=xml";
        assertEquals(expected, result);
    }
}
