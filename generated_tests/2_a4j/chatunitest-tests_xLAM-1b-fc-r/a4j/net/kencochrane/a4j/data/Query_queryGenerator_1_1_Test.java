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

public class Query_queryGenerator_1_1_Test {

    @Test
    public void testQueryGenerator() {
        Query query = new Query();
        query.serverURL = "http://example.com";
        query.associatesID = "123456";
        query.searchType = "product";
        query.type = "xml";
        query.page = "1";
        query.offer = "100";
        ArrayList<String> searchValues = new ArrayList<>();
        searchValues.add("apple");
        searchValues.add("banana");
        String expectedUrl = "http://example.com?t=123456&dev-t=DSB0XDDW1GQ3S&product=apple,banana&offerpage=1&offer=100&f=xml";
        assertEquals(expectedUrl, query.queryGenerator(query.searchType, query.type, query.page, query.offer, searchValues));
    }
}
