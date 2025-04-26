package net.kencochrane.a4j.data;

import java.util.ArrayList;
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
import java.util.Properties;

public class Query_queryGenerator_1_0_Test {

    @Test
    public void testQueryGenerator() {
        String serverURL = "http://example.com";
        String associatesID = "123";
        String token = "DSB0XDDW1GQ3S";
        String searchType = "search";
        ArrayList<String> searchValues = new ArrayList<>();
        searchValues.add("value1");
        searchValues.add("value2");
        String type = "type1";
        String page = "1";
        String offer = "offer1";
        String expectedQuery = "http://example.com?t=123&dev-t=DSB0XDDW1GQ3S&search=value1,value2&type=type1&offerpage=1&offer=offer1&f=xml";
        Query query = new Query();
        query.serverURL = serverURL;
        query.associatesID = associatesID;
        query.token = token;
        query.searchValues = searchValues;
        String actualQuery = query.queryGenerator(searchType, type, page, offer, searchValues);
        assertEquals(expectedQuery, actualQuery);
    }
}
